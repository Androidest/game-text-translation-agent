#%%
from translation_agent.agent import *
from utils import ParallelSheetChunkDispatcher, Sheet, PATH_DATA, capitalize_first_char
import json
from p3_term_retrieval import TermRetriever
from pathlib import Path
from p4_RAG import RAG
import pdb

class TermExtractorChunkDispatcher(ParallelSheetChunkDispatcher):
    use_rag = True
    RAG_MIN_COSINE_SIMILARITY = 0.97

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.term_retriever = TermRetriever()
        self.rag = RAG()
        self.first_col = self.input_sheet.column_names()[0]

    def input_chunk_to_agent(self, chunk_index:tuple) -> StructuredValidatingState:
        # prepare input
        start, end = chunk_index
        cn_dict = {}
        term_dict = {}
        for i, r in enumerate(range(start, end)):
            cn = self.input_sheet[r, self.first_col]
            terms = self.term_retriever.retrieve(cn)
            cn_dict[f"{i}"] = cn
            term_dict.update(terms)

        # RAG
        rag_translations = None
        rag_result = None
        cos_sim = None
        if self.use_rag:
            rag_result, cos_sim = self.rag.retrieve_sim(list(cn_dict.values()))
            rag_translations = rag_result[cos_sim >= self.RAG_MIN_COSINE_SIMILARITY]

        input_obj = TranslationInputSchema(
            rag_translations=rag_translations,
            term_dict=term_dict,
            cn_dict=cn_dict,
            rag_result=rag_result,
            cos_sim=cos_sim,
        )

        return StructuredValidatingState(
            input_obj=input_obj,
            input_text=input_obj.to_prompt(),
        )
    
    def output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]:
        start, end = chunk_index
        cn_dict = state["input_obj"].cn_dict
        rag_result = state["input_obj"].rag_result
        cos_sim = state["input_obj"].cos_sim
        es_dict = state["output_obj"].translations

        values = []
        for i in range(start, end):
            index = i - start
            index_key = f"{index}"
            if self.input_sheet[i, self.first_col] != cn_dict[index_key]:
                raise ValueError(f"CN not match: {self.input_sheet[i, self.first_col]} != {cn_dict[index_key]}")

            cn = cn_dict[index_key]
            es = es_dict[index_key]
            es = capitalize_first_char(es)
            terms = json.dumps(self.term_retriever.retrieve(cn), ensure_ascii=False)
            rag_text = ""
            if rag_result and cos_sim[index] >= self.RAG_MIN_COSINE_SIMILARITY:
                rag_text = f"[cos_sim={cos_sim[index]:.3f}] {rag_result[index][0]} -> {rag_result[index][1]}"

            values.append({ "CN": cn, "ES": es, "TERMS": terms, "RAG": rag_text })
        return values

def translate(data_path:str, on_update:callable = None):
    INPUT_PATH = Path(data_path)
    OUTPUT_PATH = INPUT_PATH.parent / INPUT_PATH.name.replace(".xlsx", ".translated.xlsx")

    agent = create_translation_agent(
        model=deepseek, 
        max_attempts=5
    )
    input_sheet = Sheet(
        excel_file_path=INPUT_PATH
    )
    output_sheet = Sheet(
        excel_file_path=OUTPUT_PATH,
        conlumns=["CN", "ES", "TERMS", "RAG"],
        clear=False
    )
    dispatcher = TermExtractorChunkDispatcher(
        chunk_size=5,
        parallel_chunks=10,
        input_sheet=input_sheet,
        output_sheet=output_sheet,
        agent=agent,
        desc="Translating"
    )
    dispatcher.run(on_update=on_update)
    return output_sheet

if __name__ == "__main__":
    translate(PATH_DATA / "test.xlsx", on_update=lambda dispatcher: print("Updating"))
