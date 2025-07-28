#%%
from translation_agent.agent import *
from utils import *
import json
from p3_term_retrieval import TermRetriever

class TermExtractorChunkDispatcher(ParallelSheetChunkDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.term_retriever = TermRetriever()
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

        input_obj = TranslationInputSchema(
            cn_dict=cn_dict,
            term_dict=term_dict,
        )

        return StructuredValidatingState(
            input_obj=input_obj,
            input_text=input_obj.to_prompt(),
        )
    
    def output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]:
        start, end = chunk_index
        cn_dict = state["input_obj"].cn_dict
        es_dict = state["output_obj"].translations

        values = []
        for i in range(start, end):
            index_key = f"{i-start}"
            if self.input_sheet[i, self.first_col] != cn_dict[index_key]:
                raise ValueError(f"CN not match: {self.input_sheet[i, self.first_col]} != {cn_dict[index_key]}")

            cn = cn_dict[index_key]
            es = es_dict[index_key]
            terms = json.dumps(self.term_retriever.retrieve(cn), ensure_ascii=False)
            values.append({ "CN": cn, "ES": es, "TERMS": terms })
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
        default_data={"CN": [], "ES":[], "TERMS": []},
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
