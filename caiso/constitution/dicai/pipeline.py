"""Democratic ICAI procedure adapted for CAISO system and block preferences.

Independent pairs and reasoning requests overlap; dependent debate turns stay ordered.
"""

import random
import shutil
from importlib.resources import files
from pathlib import Path
from time import monotonic

import numpy as np

from caiso.constitution.models import Constitution
from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.parallel import ordered_map
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest, read_json, save_json
from caiso.method.config import validate_method
from caiso.providers.base import check_dicai_provider
from caiso.prompts import catalog_entries, validate_resources

from .abstraction import abstract_clusters
from .activity import InductionActivity
from .clustering import cluster_principles
from .debate import debate, principle_key
from .personas import assign_personas
from .plotting import save_silhouette_artifacts
from .reasoning_assembly import assemble
from .request_cache import RequestCache
from .reconstruction import minimize_constitution, validate_constitution

PROTOCOL = "democratic-icai-arxiv-2606.28294v2-trajectory-prompts-v2"
INDUCTION_METHOD_PROMPTS = ("preferences.labeled_pair", "providers.format_retry")


def implementation_fingerprint():
    root = files("caiso.constitution.dicai")
    return digest(
        {
            p.name: p.read_text()
            for p in sorted(root.iterdir(), key=lambda p: p.name)
            if p.name.endswith(".py")
        }
    )


class DICAI:
    def __init__(self, providers, embeddings, personas, config=None, role="system_constitution"):
        self.providers, self.embeddings, self.identities = providers, embeddings, personas
        self.config, self.role = config or {}, role
        validate_method({"dicai": self.config})
        check_dicai_provider("system_constitution", getattr(providers, "roles", {}).get(role, role))

    def induce(self, pairs, seed, scope="system", progress=None, output_dir=None, activity=None):
        if not pairs:
            raise ValueError("Constitution induction requires training preference pairs")
        if len({p.pair_id for p in pairs}) != len(pairs):
            raise ValueError("DICAI preference pair IDs must be unique")
        validate_resources({"dicai": None, "method": INDUCTION_METHOD_PROMPTS})
        started = monotonic()
        workers = self.config.get("max_workers", 4)
        reasoning_workers = self.config.get("reasoning_workers", 4)
        provider = self.providers.get(self.role)
        provider_info = {
            "alias": getattr(self.providers, "roles", {}).get(self.role, self.role),
            "deployment": getattr(provider, "deployment", None),
        }
        monitor = InductionActivity(
            pairs,
            workers,
            reasoning_workers,
            (lambda state: activity({**state, "provider": provider_info})) if activity else None,
        )
        rounds = self.config.get("debate_rounds", 4)
        votes = self.config.get("reconstruction_votes", 1)
        temperatures = self.config.get("temperatures", {})
        implementation = implementation_fingerprint()
        # Settings, evidence, source procedure and prompts all invalidate pair caches.
        provenance = digest(
            {
                "protocol": PROTOCOL,
                "implementation": implementation,
                "pairs": pairs,
                "seed": seed,
                "scope": scope,
                "personas": self.identities,
                "config": self.config,
                "role": self.role,
                "provider": self.providers.fingerprint(),
                "embeddings": self.embeddings.fingerprint(),
                "prompts": catalog_entries("dicai"),
                "method_prompts": catalog_entries("method", INDUCTION_METHOD_PROMPTS),
            }
        )
        artifacts = Path(output_dir) / provenance if output_dir else None
        providers = RequestCache(self.providers, artifacts / "requests" if artifacts else None)
        if artifacts:
            save_json(
                artifacts / "manifest.json",
                {
                    "provenance": provenance,
                    "protocol": PROTOCOL,
                    "scope": scope,
                    "seed": seed,
                    "pair_ids": [p.pair_id for p in pairs],
                },
            )

        def process_pair(item):
            i, pair = item
            path = artifacts / "pairs" / (digest(pair.pair_id) + ".json") if artifacts else None
            if path and path.exists():
                record = read_json(path)
                if record["provenance"] != provenance:
                    raise ValueError("DICAI pair checkpoint provenance mismatch")
                if record["result_fingerprint"] != digest(record["result"]):
                    raise ValueError("DICAI pair checkpoint is corrupted; remove it and retry")
                monitor.update(i, "cached")
                return record["result"], True
            monitor.update(i, "reasoning")
            personas = assign_personas(self.identities, seed + i)
            catalog = assemble(
                pair, personas, providers, self.role, seed + i * 1000, self.config
            )
            values, transcript = debate(
                pair,
                catalog,
                providers,
                self.role,
                seed + i * 1000 + 500,
                rounds,
                self.config,
                progress=lambda cycle, turn: monitor.update(
                    i, "debate", cycle=cycle, total_cycles=rounds, turn=turn
                ),
            )
            result = {"principles": values, "transcript": transcript, "catalog": catalog}
            if path:
                save_json(
                    path,
                    {
                        "provenance": provenance,
                        "pair_id": pair.pair_id,
                        "pair": pair,
                        "result": result,
                        "result_fingerprint": digest(result),
                    },
                )
            monitor.update(i, "completed")
            return result, False

        def induce_pair(item):
            try:
                return process_pair(item)
            except ContentFilteredError as error:
                record_skip(
                    self.providers, error, "constitution_induction", pair_id=item[1].pair_id
                )
                monitor.update(item[0], "skipped_content_filter")
                return None
            except Exception:
                monitor.update(item[0], "failed")
                raise

        results = ordered_map(
            induce_pair,
            enumerate(pairs),
            workers,
            (lambda done, total: progress("reasoning_debate", done, total)) if progress else None,
        )
        debate_seconds = monotonic() - started
        skipped_pair_ids = [
            pair.pair_id for pair, result in zip(pairs, results, strict=True) if result is None
        ]
        retained = [
            (pair, result)
            for pair, result in zip(pairs, results, strict=True)
            if result is not None
        ]
        if not retained:
            raise ModelResponseError(
                "No unfiltered preference pairs remain for constitution induction"
            )
        pairs, results = zip(*retained, strict=True)
        settled, transcripts, catalogues, origins = [], [], [], []
        reused, expected_calls = 0, 0
        for pair, (record, cached) in zip(pairs, results, strict=True):
            values, transcript, catalog = (
                record[k] for k in ("principles", "transcript", "catalog")
            )
            settled.extend(values)
            origins.extend([pair.pair_id] * len(values))
            catalogues.append(catalog)
            transcripts.append(transcript)
            reused += int(cached)
            if not cached:
                expected_calls += sum(e["calls_before_retries"] for e in catalog)
                expected_calls += sum(
                    1
                    for e in transcript
                    if e["speaker"].startswith("debater_") or e["speaker"] == "chairperson"
                )
                expected_calls += sum(
                    len(e["result"]["votes"])
                    for e in transcript
                    if e["speaker"] == "reconstruction"
                )
        if progress:
            progress("embedding_clustering", 0, 1)
        cluster_started = monotonic()
        vectors = (
            self.providers.recovery.call(
                lambda attempt, previous: self.embeddings.embed(settled),
                stage="constitution_embeddings",
            )
            if len(settled) > 2
            else np.eye(len(settled))
        )
        cfg = self.config.get("clustering", {})
        labels, clustering = cluster_principles(
            vectors,
            cfg.get("min_clusters", 2),
            cfg.get("max_clusters", 12),
            seed,
            fixed_clusters=cfg.get("fixed_clusters"),
            cluster_step=cfg.get("cluster_step", 1),
        )
        if artifacts:
            save_silhouette_artifacts(artifacts, clustering)
            if scope == "system":
                # Keep the latest plot directly in .internal/dicai/system too.
                # The provenance directory above retains each individual induction.
                for name in ("silhouette.png", "silhouette_scores.json", "silhouette_scores.csv"):
                    latest = Path(output_dir) / name
                    temporary = latest.with_suffix(latest.suffix + ".tmp")
                    shutil.copyfile(artifacts / name, temporary)
                    temporary.replace(latest)
        cluster_seconds = monotonic() - cluster_started
        abstract_started = monotonic()
        abstraction_calls = len(set(labels))
        abstractions = abstract_clusters(
            settled,
            labels,
            providers,
            self.role,
            seed + 900000,
            max_workers=workers,
            progress=(lambda done, total: progress("abstraction", done, total))
            if progress
            else None,
            temperature=temperatures.get("abstraction", 0.2),
            config=self.config,
        )
        # Collapse equal abstraction text and combine support, exactly as upstream.
        rows, merged, cluster_mapping = [], {}, {}
        for cluster, text in enumerate(abstractions):
            members = [p for p, label in zip(settled, labels, strict=True) if label == cluster]
            key = principle_key(text)
            if key not in merged:
                merged[key] = len(rows)
                rows.append(
                    {
                        "principle": text,
                        "members": [],
                        "support": 0,
                        "medoid": settled[clustering["medoid_indices"][cluster]],
                        "silhouette": clustering["silhouette_scores"].get(
                            str(clustering["selected_k"])
                        ),
                    }
                )
            index = merged[key]
            rows[index]["members"].extend(members)
            rows[index]["support"] += len(members)
            cluster_mapping[cluster] = index
        labels = [cluster_mapping[label] for label in labels]
        outputs = [r["principle"] for r in rows]
        abstract_seconds = monotonic() - abstract_started
        if progress:
            progress("reconstruction", 0, len(pairs))
        reconstruction_started = monotonic()
        report = validate_constitution(
            providers,
            self.role,
            pairs,
            outputs,
            votes,
            temperatures.get("reconstruction", 0.0),
            progress=(lambda done, total: progress("reconstruction", done, total))
            if progress
            else None,
            max_workers=workers,
        )
        minimization = self.config.get("minimization", {})
        dropped = []
        if minimization.get("enabled", False):
            subset = pairs
            size = minimization.get("sample_size", 0)
            if size and size < len(pairs):
                subset = random.Random(seed).sample(pairs, size)
            outputs, dropped = minimize_constitution(
                providers,
                self.role,
                subset,
                outputs,
                votes,
                minimization.get("strict", False),
                temperatures.get("reconstruction", 0.0),
                max_workers=workers,
            )
            report = validate_constitution(
                providers,
                self.role,
                pairs,
                outputs,
                votes,
                temperatures.get("reconstruction", 0.0),
                max_workers=workers,
            )
        if progress:
            progress("reconstruction", len(pairs), len(pairs))
        constitution = Constitution(
            principles=outputs,
            source_preference_pair_ids=[p.pair_id for p in pairs],
            settled_principles=settled,
            cluster_ids=labels,
            abstraction_outputs=abstractions,
            metadata={
                "scope": scope,
                "seed": seed,
                "config": self.config,
                "role": self.role,
                "provider_fingerprint": self.providers.fingerprint(),
                "provider": provider_info,
                "embedding_fingerprint": self.embeddings.fingerprint(),
                "clustering": clustering,
                "clusters": rows,
                "retained_cluster_ids": [
                    i for i, row in enumerate(rows) if row["principle"] in outputs
                ],
                "principle_pair_ids": origins,
                "reasoning_catalogues": catalogues,
                "debate_transcripts": transcripts,
                "source": "DICAI arXiv:2606.28294v2 Appendix G; third_party/democratic_icai/SOURCE.json",
                "protocol": PROTOCOL,
                "provenance": provenance,
                "implementation_fingerprint": implementation,
                "reconstruction": report,
                "minimization": dropped,
                "timings": {
                    "reasoning_debate_seconds": debate_seconds,
                    "embedding_clustering_seconds": cluster_seconds,
                    "abstraction_seconds": abstract_seconds,
                    "reconstruction_seconds": monotonic() - reconstruction_started,
                    "total_seconds": monotonic() - started,
                },
                "abstraction_calls_before_retries": abstraction_calls,
                "final_reconstruction_calls_before_retries": len(pairs) * votes,
                "preference_records": len(pairs),
                "skipped_content_filter_pair_ids": skipped_pair_ids,
                "reused_pair_records": reused,
                "request_cache_hits": providers.hits,
                "model_requests_before_retries": providers.requests,
                "reasoning_debate_calls_before_retries": expected_calls,
                "max_workers": workers,
                "reasoning_workers": reasoning_workers,
                "concurrency": monitor.snapshot(),
            },
        )
        if artifacts:
            save_json(artifacts / "constitution.json", constitution)
        return constitution
