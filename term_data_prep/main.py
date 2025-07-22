#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utils import *
import time
from tqdm import tqdm
import json
from langchain.callbacks import get_openai_callback
from agent import deepseek, create_term_extractor_agent, TermExtractionState, terms_array_to_json

if __name__ == "__main__":
    agent = create_term_extractor_agent(
        model=deepseek, 
        max_retry_count=5
    )
    data_sheet = Sheet(
        excel_file_path="../data/game_lang_dataset.xlsx"
    )
    terms_sheet = Sheet(
        excel_file_path="../data/terms_extractor_dataset.xlsx",
        default_data={"CN": [], "TERMS": [], "ES":[]},
        clear=False
    )

    start = len(terms_sheet)
    end = len(data_sheet)
    batch_size = 10
    total_input_tokens = 0
    total_output_tokens = 0
    total_retry_count = 0
    for i in tqdm(range(start, end, batch_size), desc=f"Extracting terms from {start} to {end}"):
        print(f"====== Batch {i} - {i+batch_size} starts ======")
        # prepare input_list
        input_list = []
        es_list = []
        batch_str_len = 0
        for j in range(min(batch_size, end-i)):
            cn = data_sheet[i+j, "CN"]
            es = data_sheet[i+j, "ES"]
            if isinstance(cn, str):
                batch_str_len += len(cn)
                input_list.append(cn)
                es_list.append(es)
        
        print("batch_str_len:", batch_str_len)

        # extract terms, show the extraction process
        with get_openai_callback() as cb:
            state = TermExtractionState(input_list=input_list)
            for step in agent.stream(state):
                node, state_update = next(iter(step.items()))
                print(f"Node: [{node}] State:", state_update)
                state.update(state_update)
            
            print("Cached tokens", cb.prompt_tokens_cached)
            print("Input tokens", cb.prompt_tokens)
            print("Output tokens", cb.completion_tokens) 
            total_input_tokens += cb.prompt_tokens
            total_output_tokens += cb.completion_tokens
            total_retry_count += state["retry_count"]
        
        # validate the extraction
        if not state.get("output_list"):
            print(f"Unable to extract terms from line ({i} - {i+batch_size})", end="\n")
            print("Input list:\n", json.dumps(input_list, indent=4, ensure_ascii=False), end="\n")
            print("Wrong output list:\n", terms_array_to_json(state["last_response"]), end="\n")
            break

        # save to terms_sheet
        output_list = state["output_list"]
        for j in range(len(input_list)):
            cn = input_list[j]
            es = es_list[j]
            terms = output_list[j]
            terms_sheet[i+j] = { "CN": cn, "TERMS": f"{terms}", "ES": es }
        
        print("Saving Term Sheet...")
        terms_sheet.save()
        print("Total input tokens so far:", total_input_tokens)
        print("Total output tokens so far:", total_output_tokens)
        print("Total retry count so far:", total_retry_count)
        print(f"====== Batch {i} - {i+batch_size} done ======")
        time.sleep(0.3)