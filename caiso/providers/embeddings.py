import numpy as np

from caiso.core.serialization import digest


class AzureEmbeddings:
    def __init__(self, provider, batch_size=128):
        self.provider, self.batch_size = provider, batch_size

    def embed(self, texts):
        rows = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self.provider.embed(texts[i : i + self.batch_size]))
        return np.asarray(rows, dtype=float)

    def fingerprint(self):
        return self.provider.fingerprint()


class SentenceTransformerEmbeddings:
    def __init__(self, model_id, budget, device=None):
        self.model_id, self.budget, self.device = model_id, budget, device
        self.model = None

    def embed(self, texts):
        self.budget.call(local=True, embedding=True)
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_id, device=self.device)
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def fingerprint(self):
        return digest({"sentence_transformer": self.model_id})
