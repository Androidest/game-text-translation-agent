#%%
# Why use a custom graph instead of prebuild ReAct agent?
# The ReAct agent has more overhead in terms of token and time consumption:
# 1. Extra LLM calling step after a validation success. The agent must explicitly choose to finish. 
#    There's no direct path from validation tool to final output.
# 2. Full history of messages accumulates exponentially when multiple failures and attempts are encountered.
# 3. Extra context overhead for tool calling.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import StateNode, StateT, InputT, OutputT
from langchain_core.messages import *
from langchain_core.language_models import LanguageModelLike
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from typing import List, Generic
from typing_extensions import TypedDict, List, Union, Dict

class StructuredValidatingState(TypedDict):
    input_obj: Union[Dict, List, object]
    input_text: str
    output_obj: List[List[str]] #structured output: parsed object mode
    output_text: str #structured output: raw text mode
    messages: List # message history
    error: str
    attempts: int

def create_structured_validating_agent(
        model:LanguageModelLike, 
        sys_prompt:str,
        fix_error_prompt:str, # string must contain "...{error}..."
        output_schema:Generic[StateT, InputT, OutputT],
        validation:StateNode[StructuredValidatingState],
        max_attempts:int=3,
        agent_name:str="",
    ):

    llm_structured = model.with_structured_output(
        schema=output_schema,
        method="json_mode",
        include_raw=True,
    )

    def generate_node(state: StructuredValidatingState):
        if not state.get("input_text"):
            raise ValueError("input_text is None")
        
        try:
            # First llm call
            if not state.get("error"):
                messages =  [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=state["input_text"]),
                ]
                output = llm_structured.invoke(messages)
                # record the history
                messages.append(output["raw"])
                attempts = 0

            # retry llm call to fix error
            else:
                messages = [
                    *state["messages"],
                    HumanMessage(content=fix_error_prompt.format(error=state["error"])),
                ]
                output = llm_structured.invoke(messages)
                # replace the last output and error message with the new output.
                messages = messages = [
                    *state["messages"][:-1],
                    output["raw"],
                ]
                attempts = state.get("attempts", 0) + 1

            return StructuredValidatingState(
                messages = messages,
                output_obj = output["parsed"],
                output_text = output["raw"].content,
                attempts = attempts,
                error=None,
            )

        except Exception as e:
            # JSON parse error
            return StructuredValidatingState(
                messages = messages,
                output_obj = None,
                output_text = None,
                error = str(e), 
                attempts = state.get("attempts", 0)
            )

    def validate_node(state: StructuredValidatingState):
        if state.get("error"):
            return StructuredValidatingState(
                error = state["error"], # JSON parse error
            )
        return validation(state)
    
    def check_pass_condition(state: StructuredValidatingState):
        if not state.get("error"):
            return "end"
        
        if state["attempts"] >= max_attempts:
            return "end"
        
        return "retry"

    # --- nodes ---
    graph = StateGraph(StructuredValidatingState)
    graph.add_node("generate_node", generate_node)
    graph.add_node("validate_node", validate_node)
    # --- edges ---
    graph.add_edge(START, "generate_node")
    graph.add_edge("generate_node", "validate_node")
    graph.add_conditional_edges(
        "validate_node",
        check_pass_condition,
        {
            "end": END,
            "retry": "generate_node"
        }
    )
    # --- compile ---
    return graph.compile(name=agent_name)

def display_graph(graph:StateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))

# Test
if __name__ == "__main__":
    from llms import deepseek
    import asyncio
    import time

    class PersonInfoSchema(BaseModel):
        name:str = Field(description="The name of the person")
        age:int = Field(description="The age of the person")

    def validate_info(state:StructuredValidatingState):
        input_text = state["input_text"]
        person = state["output_obj"]

        if person.name not in input_text:
            return StructuredValidatingState(
                error=f"Extracted name={person.name} not in input_text",
            )
        if str(person.age) not in input_text:
            return StructuredValidatingState(
                error=f"Extracted age={person.age} not in input_text",
            )
        
        return StructuredValidatingState(
            error=None,
        )

    agent = create_structured_validating_agent(
        model=deepseek,
        sys_prompt="You are a helpful personal info extractor, extract a person's name and age from the input text.",
        fix_error_prompt="Fix the error:{error}",
        output_schema=PersonInfoSchema,
        validation=validate_info,
        max_attempts=1,
        agent_name="info_extractor_agent",
    )
    # display_graph(agent)

    async def async_run(list):
        async def async_agent(t):
            result = await agent.ainvoke(StructuredValidatingState(input_text=t))
            print("text", t)
            print("name", result["output_obj"].name, "age", result["output_obj"].age)
        
        tasks = [ async_agent(t) for t in list]
        return await asyncio.gather(*tasks)
    
    input_text = [
        "My name is John and I am 30 years old.", 
        "My name is Alice and I am 25 years old.",
        "My name is Bob and I am 40 years old.",
        "My name is Carol and I am 35 years old."
    ]
    
    start = time.time()
    [agent.invoke(StructuredValidatingState(input_text=t)) for t in input_text]
    end = time.time()
    print("Time cost:", end-start)

    start = time.time()
    asyncio.run(async_run(input_text))
    end = time.time()
    print("Time cost:", end-start)
    
