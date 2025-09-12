#%%
from p1_term_extraction.agent.agent import *
from utils import *
import json

INPUT_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"
OUTPUT_PATH = PATH_DATA / "term_extraction.xlsx"

class TermExtractorChunkDispatcher(ParallelSheetChunkDispatcher):
    def input_chunk_to_agent(self, chunk_index:tuple) -> StructuredValidatingState:
        # prepare input_list
        input_list = []
        for i in range(chunk_index[0], chunk_index[1]):
            input_list.append(self.input_sheet[i, "CN"])

        input_text=json.dumps(input_list, ensure_ascii=False)
        return StructuredValidatingState(
            input_obj=input_list,
            input_text=input_text,
        )
    
    def output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]:
        start, end = chunk_index
        output_list = state["output_obj"].terms
        values = []
        for i in range(start, end):
            cn = self.input_sheet[i, "CN"]
            es = self.input_sheet[i, "ES"]
            terms = json.dumps(output_list[i-start], ensure_ascii=False)
            values.append({ "CN": cn, "TERMS": f"{terms}", "ES": es })
        return values

if __name__ == "__main__":
    agent = create_term_extraction_agent(
        model=default_llm, 
        max_attempts=5
    )
    input_sheet = Sheet(
        excel_file_path=INPUT_PATH
    )
    output_sheet = Sheet(
        excel_file_path=OUTPUT_PATH,
        default_data={"CN": [], "TERMS": [], "ES":[]},
        clear=False
    )
    dispatcher = TermExtractorChunkDispatcher(
        chunk_size=10,
        parallel_chunks=20,
        input_sheet=input_sheet,
        output_sheet=output_sheet,
        agent=agent,
        desc="Extracting terms"
    )
    dispatcher.run()
