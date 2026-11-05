import os
import dspy
import os.path as osp
from dotenv import load_dotenv

from caiso.arch.system import CompoundAISystem
from caiso.prompts import entry as prompt_entry
from caiso.arch.base import BaseComponent
from caiso.adapt.dspy import create_component_from_dspy
from examples.metrics.f1_score import f1_score


class QuestionRewriter(dspy.Signature):
    __doc__ = prompt_entry('systems/hotpotqa', 'QuestionRewriter.instructions')
    question: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'QuestionRewriter.question.prefix')
    )
    rewritten_query: str = dspy.OutputField(
        prefix=prompt_entry('systems/hotpotqa', 'QuestionRewriter.rewritten_query.prefix')
    )

class InfoExtractor(dspy.Signature):
    __doc__ = prompt_entry('systems/hotpotqa', 'InfoExtractor.instructions')

    rewritten_query: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'InfoExtractor.rewritten_query.prefix')
    )
    search_keywords: str = dspy.OutputField(
        prefix=prompt_entry('systems/hotpotqa', 'InfoExtractor.search_keywords.prefix')
    )


class WikipediaRetriever(BaseComponent):
    def __init__(self, k, variable_search_space):
        super().__init__(
            description="Retrieve content from Wikipedia.",
            input_fields=["search_keywords"],
            output_fields=["retrieve_content"],
            variable={"k": k},
            variable_search_space=variable_search_space
        )
        load_dotenv('.env')
        colbert_url = os.getenv('COLBERT_URL') or os.getenv('colbert_url') or 'http://20.102.90.50:2017/wiki17_abstracts'
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(url=colbert_url)
        dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

    def forward(self, **inputs):
        search_keywords = inputs.get("search_keywords")
        if not search_keywords:
            raise ValueError("Missing required input: 'search_keywords'")

        topk_passages = dspy.Retrieve(k=self.variable["k"])(search_keywords).passages
        retrieve_content = "\n".join(topk_passages)
        return {"retrieve_content": retrieve_content}


class HintGenerator(dspy.Signature):
    __doc__ = prompt_entry('systems/hotpotqa', 'HintGenerator.instructions')

    rewritten_query: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'HintGenerator.rewritten_query.prefix')
    )
    retrieve_content: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'HintGenerator.retrieve_content.prefix')
    )
    hints: str = dspy.OutputField(
        prefix=prompt_entry('systems/hotpotqa', 'HintGenerator.hints.prefix')
    )


class AnswerGenerator(dspy.Signature):
    __doc__ = prompt_entry('systems/hotpotqa', 'AnswerGenerator.instructions')
    rewritten_query: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'AnswerGenerator.rewritten_query.prefix')
    )
    hints: str = dspy.InputField(
        prefix=prompt_entry('systems/hotpotqa', 'AnswerGenerator.hints.prefix')
    )
    answer: str = dspy.OutputField(prefix=prompt_entry('systems/hotpotqa', 'AnswerGenerator.answer.prefix'))


def system_engine(*args, **kwargs):
    lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=1024, temperature=0.6)
    dspy.settings.configure(lm=lm)

    # System automatically logs summary and Mermaid graph on initialization
    system = CompoundAISystem(
        components={
            "question_rewriter": create_component_from_dspy(QuestionRewriter),
            "info_extractor": create_component_from_dspy(InfoExtractor),
            "wikipedia_retriever": WikipediaRetriever(k=1, variable_search_space={"k": [1, 5, 10, 25]}),
            "hint_generator": create_component_from_dspy(HintGenerator),
            "answer_generator": create_component_from_dspy(AnswerGenerator),
        },
        final_output_fields=["answer"],
        ground_fields=["gd_answer"],
        eval_func=f1_score,
        *args, **kwargs
    )
    
    return system


if __name__ == "__main__":

    from examples.datasets.hotpotqa import dataset_engine
    trainset, valset, testset = dataset_engine()
    dotenv_path = osp.expanduser('.env')
    load_dotenv(dotenv_path)

    system = system_engine()

    with system.components['wikipedia_retriever'].context(variable={"k": 50}):
        score = system.evaluate(testset[0])
        print(score)

    with system.components['question_rewriter'].context(model="openai/gpt-4o", max_tokens=1024):
        prediction = system(**testset[0])
        print(prediction)
        print('prediction:', prediction.answer, 'answer:', testset[0].gd_answer)
        score = system.evaluate(testset[0], prediction)
        print(score)
                


