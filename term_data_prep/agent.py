#%%
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import *
from langchain_core.language_models import LanguageModelLike
from langchain.callbacks import get_openai_callback
from IPython.display import Image, display
from promtps import *
from state import *
from llms import *
import json
from pydantic import BaseModel, Field
from typing import List

class Extraction(BaseModel):
    terms: List[List[str]] = Field(description="每一行的术语列表")

def create_term_extractor_agent(model:LanguageModelLike, max_retry_count=3):
    llm_structured = model.with_structured_output(
        schema=Extraction,
        method="json_mode",
    )

    def extract_terms_node(state: TermExtractionState):
        if not state.get("input_list"):
            raise ValueError("input_list is None")
        
        input_text = json.dumps(state["input_list"], indent=1, ensure_ascii=False)

        # First try of extraction
        if not state.get("error"):
            messages =  [
                SystemMessage(content=TERM_EXTRACTOR_INSTRUCTION),
                HumanMessage(content=input_text),
            ]
            response = llm_structured.invoke(messages)
            # record the extraction
            messages.append(
                AIMessage(content=terms_array_to_json(response.terms))
            )
            retry_count = 0

        # if error, try to fix it
        else:
            messages = [
                *state["messages"],
                HumanMessage(content=FIX_EXTRACTION_INSTRUCTION.format(error=state["error"])),
            ]
            response = llm_structured.invoke(messages)
            # drop the last extraction and error message, and replace them with the extraction
            messages = messages = [
                *state["messages"][:-1],
                AIMessage(content=terms_array_to_json(response.terms)),
            ]
            retry_count = state["retry_count"] + 1

        return TermExtractionState(
            messages = messages,
            last_response = response.terms,
            retry_count = retry_count,
        )

    def validate_node(state: TermExtractionState):
        input_list = state["input_list"]
        output_list = state["last_response"]
        
        # check output length
        if len(output_list) != len(input_list):
            return TermExtractionState(
                error = ERROR_EXTRACTION_LENGTH.format(
                    input_length=len(input_list),
                    output_length=len(output_list)
                ),
            )
        
        # check if the extracted terms are substrings of the input text
        for i in range(len(input_list)):
            text = input_list[i]
            for term in output_list[i]:
                if term not in text:
                    return TermExtractionState(
                        error = ERROR_EXTRACTION_LINE.format(
                            line=i,
                            text=text,
                            wrong_term=term
                        )
                    )

        return TermExtractionState(
            output_list=output_list,
            error=None
        )
    
    def check_pass_condition(state: TermExtractionState):
        if state.get("output_list"):
            return "success"
        
        if state["retry_count"] >= max_retry_count:
            return "fail"
        
        return "retry"

    # --- nodes ---
    graph = StateGraph(TermExtractionState)
    graph.add_node("extract_terms_node", extract_terms_node)
    graph.add_node("validate_node", validate_node)
    # --- edges ---
    graph.add_edge(START, "extract_terms_node")
    graph.add_edge("extract_terms_node", "validate_node")
    graph.add_conditional_edges(
        "validate_node",
        check_pass_condition,
        {
            "fail": END,
            "success": END,
            "retry": "extract_terms_node"
        }
    )
    # --- compile ---
    return graph.compile(name="term_extract_agent")

def display_graph(graph:StateGraph):
    display(Image(graph.get_graph().draw_mermaid_png()))

def terms_array_to_json(terms:List[List[str]]):
    return f"{terms}".replace("],", "],\n")

if __name__ == "__main__":
    agent = create_term_extractor_agent(model=deepseek, max_retry_count=1)
    # display_graph(agent)
    input_list = ["#585天梯门票不足\n是否消耗{0}购买#585 1并挑战对手", "(初始)", "同伴6普通攻击", "Lv1 雨林幻境", "出战和助战的花灵，属性会直接加成到角色身上。\\n2、花灵升级需要消耗金币和花灵经验。\\n3、花灵升级到一定等级后，会需要突破才能继续升级。\\n4、每个品阶有等级上限，达到上限后，需要升阶才可以继续升级。\\n5、花灵达到一定等级后，会解锁或者升级对应技能，详情查看技能说明。"]
    state = TermExtractionState(
        input_list=input_list,
    )

    # show the extraction process, step by step, instead of a single invoke
    # state = agent.invoke(state)
    with get_openai_callback() as cb:
        for step in agent.stream(state):
            node, state = next(iter(step.items()))
            print(f"Node [{node}] State:", state)

        print("Cached", cb.prompt_tokens_cached)
        print("total_input", cb.prompt_tokens)
        print("total_output", cb.completion_tokens) 
        print("total_all", cb.total_tokens)

    # the output_list would be missing if an error is encountered
    if state.get("output_list"):
        print(terms_array_to_json(input_list))
        print(terms_array_to_json(state["output_list"]))
    else:
        print(state["error"])
