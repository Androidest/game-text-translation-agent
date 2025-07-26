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

INPUT_PATH = "../data/game_lang_dataset_cleaned.xlsx"
OUTPUT_PATH = "../data/term_extraction.xlsx"
EXIT_HOT_KEY = "esc"

if __name__ == "__main__":
    agent = create_term_extraction_agent(
        model=deepseek, 
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
    key_checker = keyStrokeListener()
    key_checker.add_hotkey(EXIT_HOT_KEY)

    start = len(output_sheet)
    end = len(input_sheet)
    batch_size = 10
    total_cached_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_attempts = 0
    progress_bar = tqdm(total=end, initial=start, desc=f"Aligning terms from {start} to {end}")

    for i in range(start, end, batch_size):
        print(f"====== Batch {i} - {i+batch_size} starts ======")
        # prepare input_list
        input_list = []
        batch_str_len = 0
        for j in range(min(batch_size, end-i)):
            cn = input_sheet[i+j, "CN"]
            input_list.append(cn)
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
                    print(f"Node: [{node}] attempt: {state.get('attempts', 0)}, error:{state_update.get("error")}")
            
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
            cn = input_sheet[i+j, "CN"]
            es = input_sheet[i+j, "ES"]
            terms = json.dumps(output_list[j], ensure_ascii=False)
            output_sheet[i+j] = { "CN": cn, "TERMS": f"{terms}", "ES": es }
        
        print("Saving Term Sheet...")
        output_sheet.save()
        progress_bar.update(len(input_list))
        print("Alignment sheet saved to: ", OUTPUT_PATH)
        print("Total cached tokens so far:", total_cached_tokens)
        print("Total input tokens so far:", total_input_tokens)
        print("Total output tokens so far:", total_output_tokens)
        print("Total attempts so far:", total_attempts)
        print(f"====== Batch {i} - {i+batch_size} done ======")

        if key_checker.has_key_pressed(f"{EXIT_HOT_KEY}"):
            print(f"Exit hotkeys was pressed. Exiting...")
            break