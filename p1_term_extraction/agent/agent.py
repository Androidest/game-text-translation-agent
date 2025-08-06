#%%
from utils import *
from p1_term_extraction.agent.promtps import *
from pydantic import BaseModel, Field
from typing import List

class TermExtractionSchema(BaseModel):
    terms: List[List[str]] = Field(description="每一条文本的术语列表")

def create_term_extraction_agent(
        model,
        max_attempts:int=3
    ):
    
    def validate_extraction(state: StructuredValidatingState):
        input_list = state["input_obj"]
        output_list = state["output_obj"].terms

        # check output length
        if len(output_list) != len(input_list):
            return StructuredValidatingState(
                error = ERROR_EXTRACTION_LENGTH.format(
                    input_length=len(input_list),
                    output_length=len(output_list)
                ),
            )
        
        # check if the extracted terms are substrings of the input_text text
        for i in range(len(input_list)):
            text = input_list[i]
            for term in output_list[i]:
                if term not in text:
                    return StructuredValidatingState(
                        error = ERROR_EXTRACTION_LINE.format(
                            line=i,
                            text=text,
                            wrong_term=term
                        )
                    )

        return StructuredValidatingState(
            error=None
        )

    return create_structured_validating_agent(
        model=model,
        sys_prompt=TERM_EXTRACTOR_INSTRUCTION,
        fix_error_prompt=FIX_ERROR_INSTRUCTION,
        output_schema=TermExtractionSchema,
        validation=validate_extraction,
        max_attempts=max_attempts,
        agent_name="term_extractor_agent",
    )

if __name__ == "__main__":
    from langchain_community.callbacks.manager import get_openai_callback
    import json

    agent = create_term_extraction_agent(
        model=deepseek,
        max_attempts=1,
    )

    input_list = ["#585天梯门票不足\n是否消耗{0}购买#585 1并挑战对手", "(初始)", "同伴6普通攻击", "Lv1 雨林幻境", "出战和助战的花灵，属性会直接加成到角色身上。\\n2、花灵升级需要消耗金币和花灵经验。\\n3、花灵升级到一定等级后，会需要突破才能继续升级。\\n4、每个品阶有等级上限，达到上限后，需要升阶才可以继续升级。\\n5、花灵达到一定等级后，会解锁或者升级对应技能，详情查看技能说明。"]
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
        print(json.dumps(state["output_obj"].terms, indent=4, ensure_ascii=False))
