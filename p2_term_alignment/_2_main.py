#%%
from p2_term_alignment.agent import *
from utils import *
import json

INPUT_PATH = PATH_DATA / "term_extraction.final.cleaned.xlsx"
OUTPUT_PATH = PATH_DATA / "term_alignment.xlsx"
EXIT_HOT_KEY = "esc"

class TermAlignmentChunkDispatcher(ParallelSheetChunkDispatcher):
    def input_chunk_to_agent(self, chunk_index:tuple) -> StructuredValidatingState:
        # prepare input_list
        input_list = []
        for i in range(chunk_index[0], chunk_index[1]):
            sample = {
                "cn": self.input_sheet[i, "CN"],
                "es": self.input_sheet[i, "ES"],
                "terms": json.loads(self.input_sheet[i, "TERMS"]),
            }
            input_list.append(sample)

        input_text=json.dumps(input_list, ensure_ascii=False)
        return StructuredValidatingState(
            input_obj=input_list,
            input_text=input_text,
        )
    
    def output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]:
        start, end = chunk_index
        output_list = state["output_obj"].alignments
        values = []
        for i in range(start, end):
            cn = self.input_sheet[i, "CN"]
            es = self.input_sheet[i, "ES"]
            terms = self.input_sheet[i, "TERMS"]
            terms_es = json.dumps(output_list[i-start], ensure_ascii=False)
            values.append({ "CN": cn, "ES": es, "TERMS": terms, "TERMS_ES": terms_es })
        return values

if __name__ == "__main__":
    agent = create_term_aligment_agent(
        model=default_llm, 
        max_attempts=5
    )
    input_sheet = Sheet(
        excel_file_path=INPUT_PATH
    )
    output_sheet = Sheet(
        excel_file_path=OUTPUT_PATH,
        default_data={"CN": [], "ES":[], "TERMS": [], "TERMS_ES":[] },
        clear=False
    )
    dispatcher = TermAlignmentChunkDispatcher(
        chunk_size=5,
        parallel_chunks=30,
        input_sheet=input_sheet,
        output_sheet=output_sheet,
        agent=agent,
        desc="Aligning terms"
    )
    dispatcher.run()