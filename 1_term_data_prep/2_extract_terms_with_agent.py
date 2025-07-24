#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utils import *
import asyncio
import time
from tqdm import tqdm
import json
from langchain.callbacks import get_openai_callback
from agent import *

if __name__ == "__main__":
    agent = create_term_extraction_agent(
        model=deepseek, 
        max_attempts=5
    )
    data_sheet = Sheet(
        excel_file_path="../data/game_lang_dataset_cleaned.xlsx"
    )
    terms_sheet = Sheet(
        excel_file_path="../data/terms_extractor_dataset.xlsx",
        default_data={"CN": [], "TERMS": [], "ES":[]},
        clear=False
    )

    start = len(terms_sheet)
    end = len(data_sheet)
    batch_size = 10
    total_cached_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_attempts = 0
    for i in tqdm(range(start, end, batch_size), desc=f"Extracting terms from {start} to {end}"):
        print(f"====== Batch {i} - {i+batch_size} starts ======")
        # prepare input_list
        input_list = []
        es_list = []
        batch_str_len = 0
        for j in range(min(batch_size, end-i)):
            cn = data_sheet[i+j, "CN"]
            es = data_sheet[i+j, "ES"]
            input_list.append(cn)
            es_list.append(es)
            batch_str_len += len(cn)
        print("batch_str_len:", batch_str_len)

        # extract terms, show the extraction process
        with get_openai_callback() as cb:
            state = StructuredValidatingState(
                input_obj=input_list,
                input_text=json.dumps(input_list, ensure_ascii=False),
            )
            print("Input text:", json.dumps(input_list, indent=4, ensure_ascii=False))
            for step in agent.stream(state):
                node, state_update = next(iter(step.items()))
                state.update(state_update)
                if node == "generate_node":
                    print(f"Node: [{node}] output: {state_update.get("output_text")}, error:{state_update.get("error")}")
                if node == "validate_node":
                    print(f"Node: [{node}] attempt: {state_update.get('attempts', 0)}, error:{state_update.get("error")}")
            
            print("Cached tokens", cb.prompt_tokens_cached)
            print("Input tokens", cb.prompt_tokens)
            print("Output tokens", cb.completion_tokens) 
            total_cached_tokens += cb.prompt_tokens_cached
            total_input_tokens += cb.prompt_tokens
            total_output_tokens += cb.completion_tokens
            total_attempts += state["attempts"]
        
        # validate the extraction
        if not state.get("output_obj"):
            print(f"Unable to extract terms from line ({i} - {i+batch_size})", end="\n")
            print("Input list:\n", state["input_text"])
            print("Wrong output list:\n", state["output_text"])
            break

        # save to terms_sheet
        output_list = state["output_obj"].terms
        for j in range(len(input_list)):
            cn = input_list[j]
            es = es_list[j]
            terms = output_list[j]
            terms_sheet[i+j] = { "CN": cn, "TERMS": f"{terms}", "ES": es }
        
        print("Saving Term Sheet...")
        terms_sheet.save()
        print("Total cached tokens so far:", total_cached_tokens)
        print("Total input tokens so far:", total_input_tokens)
        print("Total output tokens so far:", total_output_tokens)
        print("Total attempts so far:", total_attempts)
        print(f"====== Batch {i} - {i+batch_size} done ======")
        # time.sleep(0.1)