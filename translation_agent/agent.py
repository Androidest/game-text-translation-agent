#%%
from translation_agent.promtps import *
from utils import StructuredValidatingState, create_structured_validating_agent, deepseek
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import json

class TranslationInputSchema(BaseModel):
    rag_translations: Optional[List[List[str]]] = Field(description="参考类似过往翻译（RAG检索结果）")
    term_dict: Dict[str, List[str]] = Field(description="参考术语字典")
    cn_dict: Dict[str, str] = Field(description="输入中文原文字典")
    
    rag_result: Optional[List[List[str]]] = Field(description="RAG检索结果")
    cos_sim: Optional[List[float]] = Field(description="RAG检索结果的余弦相似度")

    def to_prompt(self):
        return INPUT_PROMPT_TEMPLATE.format(
            rag_translations=self.rag_text(),
            term_dict=json.dumps(self.term_dict, ensure_ascii=False),
            cn_dict=json.dumps(self.cn_dict, ensure_ascii=False),
        )
    
    def rag_text(self):
        if not self.rag_translations or len(self.rag_translations) == 0:
            rag_text = "无"
        else:
            rag_text = ""
            for src, translation in self.rag_translations:
                rag_text += f"[*] {src} -> {translation}\n"
            rag_text = rag_text.strip()
        return rag_text

class TranslationOutputSchema(BaseModel):
    translations: Dict[str, str] = Field(description="键：索引号，对应原文索引号。值：西班牙语翻译")

def create_translation_agent(
        model,
        max_attempts:int=3
    ):
    
    def validate_translation(state: StructuredValidatingState):
        cn_dict = state["input_obj"].cn_dict
        es_dict = state["output_obj"].translations

        # check output length
        if len(es_dict) != len(cn_dict):
            return StructuredValidatingState(
                error = ERROR_OUTPUT_LENGTH.format(
                    input_length=len(cn_dict),
                    output_length=len(es_dict)
                ),
            )
        
        # check index keys
        for index_key in cn_dict.keys():
            if index_key not in es_dict:
                return StructuredValidatingState(
                    error = ERROR_INDEXING.format(
                        index=index_key,
                    ),
                )
            
        # check \\n
        for index_key in cn_dict.keys():
            cn = cn_dict[index_key]
            es = es_dict[index_key]
            if cn.count("\\n") != es.count("\\n"):
                return StructuredValidatingState(
                    error = ERROR_NEXT_LINE.format(
                        line=index_key,
                        input_nl_count=cn.count("\\n"),
                        output_nl_count=es.count("\\n"),
                    ),
                )

        return StructuredValidatingState(
            error=None
        )

    return create_structured_validating_agent(
        model=model,
        sys_prompt=TRANSLATION_INSTRUCTION,
        fix_error_prompt=FIX_ERROR_INSTRUCTION,
        output_schema=TranslationOutputSchema,
        validation=validate_translation,
        max_attempts=max_attempts,
        agent_name="translation_agent",
    )

if __name__ == "__main__":
    from langchain_community.callbacks.manager import get_openai_callback

    agent = create_translation_agent(
        model=deepseek,
        max_attempts=1,
    )

    input_obj = TranslationInputSchema(
        cn_dict={
            "0": "2个相同属性的主要出战花灵可以为你的攻击附上属性", 
            "1": "(倒计时结束后点击中央柱子召唤下一组BOSS)",
        },
        term_dict={
            "出战花灵": ["Hadas Florales principales"],
            "同属性": ["con los mismos atributos"],
            "召唤": ["convocar"],
            "BOSS": ["Jefes"],
            "中央柱子": ["pilar central"],
        }
    )
    input_text = input_obj.to_prompt()
    
    state = StructuredValidatingState(
        input_text=input_text,
        input_obj=input_obj,
    )

    with get_openai_callback() as cb:
        for step in agent.stream(state):
            node, state_update = next(iter(step.items()))
            state.update(state_update)
            print(f"Node [{node}] State Update:", state_update)

        print("Cached", cb.prompt_tokens_cached)
        print("total_input", cb.prompt_tokens)
        print("total_output", cb.completion_tokens) 
        print("total_all", cb.total_tokens)

    if state.get("error"):
        print(state["output_text"])
        print(state["error"])
    else:
        print(input_text)
        print(json.dumps(state["output_obj"].translations, indent=4, ensure_ascii=False))
