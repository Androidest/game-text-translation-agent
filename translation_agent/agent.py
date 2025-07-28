#%%
from p2_term_alignment.promtps import *
from utils import *
from pydantic import BaseModel, Field
from typing import List, Dict

class TermAlignmentSchema(BaseModel):
    alignments: List[Dict[str, str]] = Field(description="数组每个元素对应每组中西平行文本的术语和对齐的西班牙语翻译")

def create_term_aligment_agent(
        model,
        max_attempts:int=3
    ):
    
    def validate_alignment(state: StructuredValidatingState):
        input_list = state["input_obj"]
        output_list = state["output_obj"].alignments

        # check output length
        if len(output_list) != len(input_list):
            return StructuredValidatingState(
                error = ERROR_OUTPUT_LENGTH.format(
                    input_length=len(input_list),
                    output_length=len(output_list)
                ),
            )
        
        for i in range(len(input_list)):
            sample = input_list[i]
            aligned_terms = output_list[i]

            # check if the number of terms are the same
            if len(sample["terms"]) != len(aligned_terms):
                return StructuredValidatingState(
                    error = ERROR_ALIGNMENT_LENGTH.format(
                        line=i,
                        input_length=len(sample["terms"]),
                        output_length=len(aligned_terms)
                    ),
                )

            # check if the extracted translation are substrings of the spanish entire spanish translation
            for term_cn, term_es in aligned_terms.items():
                if term_es not in sample["es"]:
                     return StructuredValidatingState(
                        error = ERROR_ALIGNMENT_LINE.format(
                            line=i,
                            es_text=sample["es"],
                            term=term_cn,
                            wrong_alignment=term_es
                        )
                    )

        return StructuredValidatingState(
            error=None
        )

    return create_structured_validating_agent(
        model=model,
        sys_prompt=TERM_ALIGNMENT_INSTRUCTION,
        fix_error_prompt=FIX_ERROR_INSTRUCTION,
        output_schema=TermAlignmentSchema,
        validation=validate_alignment,
        max_attempts=max_attempts,
        agent_name="term_alignment_agent",
    )

if __name__ == "__main__":
    from langchain_community.callbacks.manager import get_openai_callback
    import json

    agent = create_term_aligment_agent(
        model=deepseek,
        max_attempts=1,
    )

    input_list = [
        {   
            "cn": "2个相同属性的主要出战花灵可以为你的攻击附上属性", 
            "es": "2 Hadas Florales principales con los mismos atributos pueden agregar atributos a tus ataques", 
            "terms": ["属性", "出战花灵", "攻击", "同属性"]
        },
        {
            "cn": "(倒计时结束后点击中央柱子召唤下一组BOSS)",
            "es": "(Pulsa el pilar central al terminar la cuenta regresiva para convocar el siguiente grupo de Jefes)",
            "terms": ["点击", "召唤", "BOSS", "下", "中央柱子", "倒计时"]
        },
        {
            "cn": "本观星术士再也不能忍耐了！{0}！",
            "es": "¡No lo soporto más! ¡{0}!",
            "terms": ["观星术士", "忍耐"]
        }
    ]
    
    state = StructuredValidatingState(
        input_text=json.dumps(input_list, ensure_ascii=False),
        input_obj=input_list,
    )

    # show the extraction process, step by step, instead of a single invoke
    # state = agent.invoke(state)
    with get_openai_callback() as cb:
        for step in agent.stream(state):
            node, state_update = next(iter(step.items()))
            state.update(state_update)
            print(f"Node [{node}] State Update:", state_update)

        print("Cached", cb.prompt_tokens_cached)
        print("total_input", cb.prompt_tokens)
        print("total_output", cb.completion_tokens) 
        print("total_all", cb.total_tokens)

    # the output would be missing if an error is encountered
    if state.get("error"):
        print(state["error"])
    else:
        print(json.dumps(input_list, indent=4, ensure_ascii=False))
        print(state["output_text"])
        print(json.dumps(state["output_obj"].alignments, indent=4, ensure_ascii=False))
