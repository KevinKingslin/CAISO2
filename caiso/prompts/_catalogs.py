"""Catalog keys mapped to text resources; prompt content lives only in .txt files.

Paths are relative to this package without the .txt suffix. Tuples preserve
the order of multi-entry defaults such as prompt tips and personas.
"""

CATALOGS = {
    "method": {
        "prompt_optimizer.default_tips": (
            "method/prompt_optimizer/default_tips/01",
            "method/prompt_optimizer/default_tips/02",
            "method/prompt_optimizer/default_tips/03",
            "method/prompt_optimizer/default_tips/04",
        ),
        "default_personas": (
            "method/default_personas/01",
            "method/default_personas/02",
            "method/default_personas/03",
        ),
        "prompt_optimizer.request": "method/prompt_optimizer/prompt",
        "block_selection.request": "method/block_selection/prompt",
        "generator.request": "method/generator/prompt",
        "services.schema_message": "method/services/schema_message",
        "providers.format_retry": "method/providers/format_retry",
        "preferences.labeled_pair": "method/preferences/labeled_pair",
        "preferences.unlabeled_pair": "method/preferences/unlabeled_pair",
    },
    "systems/amazon": {
        "SessionAnalyzerModule.template": "systems/amazon/session_analyzer/prompt",
        "CandidateProfilerModule.template": "systems/amazon/candidate_profiler/prompt",
        "NextItemDecider.instructions": "systems/amazon/next_item_decider/instructions",
        "NextItemDecider.context.prefix": "systems/amazon/next_item_decider/fields/context_prefix",
        "NextItemDecider.context.desc": "systems/amazon/next_item_decider/fields/context_description",
        "NextItemDecider.feedback.prefix": "systems/amazon/next_item_decider/fields/feedback_prefix",
        "NextItemDecider.feedback.desc": "systems/amazon/next_item_decider/fields/feedback_description",
        "NextItemDecider.answer.prefix": "systems/amazon/next_item_decider/fields/answer_prefix",
        "NextItemDecider.answer.desc": "systems/amazon/next_item_decider/fields/answer_description",
    },
    "systems/hotpotqa": {
        "QuestionRewriter.instructions": "systems/hotpotqa/question_rewriter/instructions",
        "QuestionRewriter.question.prefix": "systems/hotpotqa/question_rewriter/fields/question_prefix",
        "QuestionRewriter.rewritten_query.prefix": "systems/hotpotqa/question_rewriter/fields/rewritten_query_prefix",
        "InfoExtractor.instructions": "systems/hotpotqa/info_extractor/instructions",
        "InfoExtractor.rewritten_query.prefix": "systems/hotpotqa/info_extractor/fields/rewritten_query_prefix",
        "InfoExtractor.search_keywords.prefix": "systems/hotpotqa/info_extractor/fields/search_keywords_prefix",
        "HintGenerator.instructions": "systems/hotpotqa/hint_generator/instructions",
        "HintGenerator.rewritten_query.prefix": "systems/hotpotqa/hint_generator/fields/rewritten_query_prefix",
        "HintGenerator.retrieve_content.prefix": "systems/hotpotqa/hint_generator/fields/retrieve_content_prefix",
        "HintGenerator.hints.prefix": "systems/hotpotqa/hint_generator/fields/hints_prefix",
        "AnswerGenerator.instructions": "systems/hotpotqa/answer_generator/instructions",
        "AnswerGenerator.rewritten_query.prefix": "systems/hotpotqa/answer_generator/fields/rewritten_query_prefix",
        "AnswerGenerator.hints.prefix": "systems/hotpotqa/answer_generator/fields/hints_prefix",
        "AnswerGenerator.answer.prefix": "systems/hotpotqa/answer_generator/fields/answer_prefix",
    },
    "systems/pubmed": {
        "ContextAnalystModule.instruction": "systems/pubmedqa/context_analyst/instructions",
        "ContextAnalystModule.request": "systems/pubmedqa/context_analyst/prompt",
        "ProblemSolverModule.instruction": "systems/pubmedqa/problem_solver/instructions",
        "ProblemSolverModule.request": "systems/pubmedqa/problem_solver/prompt",
    },
    "personas/amazon": {
        "E-commerce behavior analyst": "systems/amazon/personas/01_behavior_analyst",
        "Product matching specialist": "systems/amazon/personas/02_product_matching_specialist",
        "Recommendation evaluator": "systems/amazon/personas/03_recommendation_evaluator",
    },
    "personas/hotpotqa": {
        "Multi-hop question answering researcher": "systems/hotpotqa/personas/01_multi_hop_qa_researcher",
        "Information retrieval specialist": "systems/hotpotqa/personas/02_information_retrieval_specialist",
        "Fact verification specialist": "systems/hotpotqa/personas/03_fact_verification_specialist",
    },
    "personas/pubmedqa": {
        "Biomedical evidence analyst": "systems/pubmedqa/personas/01_biomedical_evidence_analyst",
        "Clinical research methodologist": "systems/pubmedqa/personas/02_clinical_research_methodologist",
        "Scientific question answering evaluator": "systems/pubmedqa/personas/03_scientific_qa_evaluator",
    },
    "dicai": {
        "reasoning.cot": "dicai/reasoning/cot",
        "reasoning.self_consistency": "dicai/reasoning/self_consistency",
        "reasoning.self_refine": "dicai/reasoning/self_refine",
        "debate.debater": "dicai/debate/debater",
        "debate.chairperson": "dicai/debate/chairperson",
        "reconstruction": "dicai/reconstruction",
        "abstraction": "dicai/abstraction",
    },
}
