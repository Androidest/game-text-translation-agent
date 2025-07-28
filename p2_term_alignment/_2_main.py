#%%
from p2_term_alignment.agent import *
from utils import *
import asyncio
import json
from langchain_community.callbacks.manager import get_openai_callback

INPUT_PATH = PATH_DATA / "term_extraction.final.cleaned.xlsx"
OUTPUT_PATH = PATH_DATA / "term_alignment.xlsx"
EXIT_HOT_KEY = "esc"

async def async_process_chunk(chunk_index:tuple, input_sheet:Sheet, agent:StateGraph):
    # prepare input_list
    input_list = []
    for i in range(chunk_index[0], chunk_index[1]):
        sample = {
            "cn": input_sheet[i, "CN"],
            "es": input_sheet[i, "ES"],
            "terms": json.loads(input_sheet[i, "TERMS"]),
        }
        input_list.append(sample)

    input_text=json.dumps(input_list, ensure_ascii=False)

    # align terms
    logs = f"====== Chunk ({chunk_index[0]} - {chunk_index[1]}) starts ======"
    logs += f"\nbatch_str_len: {len(input_text)}"
    # logs += f"\nInput text: {input_text}"

    state = StructuredValidatingState(
        input_obj=input_list,
        input_text=input_text,
    )
    async for step in agent.astream(state):
        node, state_update = next(iter(step.items())) 
        state.update(state_update)
        # if node == "generate_node":
        #     logs += f"Node: [{node}] output: {state_update.get("output_text")}, error:{state_update.get("error")}\n"
        if node == "validate_node":
            logs += f"\nNode: [{node}] attempt: {state.get('attempts', 0)}, error:{state_update.get("error")}"

    return state, logs

async def async_process_multiple_chunks(chunk_list:list, input_sheet:Sheet, agent:StateGraph):
    tasks = [async_process_chunk(chunk_index, input_sheet, agent) for chunk_index in chunk_list]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    agent = create_term_aligment_agent(
        model=deepseek, 
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
    start = len(output_sheet)
    max_lines = len(input_sheet)
    chunk_size = 5
    parallel_chunks = 30
    total_cached_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_attempts = 0

    key_checker = keyStrokeListener()
    key_checker.add_hotkey(EXIT_HOT_KEY)
    chunks_iterator = SheetParallelChunksIterator(output_sheet, max_lines, chunk_size, parallel_chunks, desc="Aligning terms")
    for chunk_list in chunks_iterator:
        print('#'*100)
        print(f"Processing chunks: {json.dumps(chunk_list, ensure_ascii=False)}")
        print(f"Note: Press '{EXIT_HOT_KEY}' to exit safely at any time.")
        
        with get_openai_callback() as cb:
            # process the chunks in parallel
            results = asyncio.run(async_process_multiple_chunks(chunk_list, input_sheet, agent))
            # process the chunks results
            for (start, end), (state, logs) in zip(chunk_list, results):
                print(logs)

                # validate the extraction
                if state.get("error"):
                    print(f"Unable to align terms from line ({start} - {end})", end="\n")
                    print("Input list:\n", state["input_text"])
                    print("Wrong output list:\n", state["output_text"])
                    print(f"========== Chunk ({start} - {end}) error ==========\n")
                # complete the chunk
                else:
                    output_list = state["output_obj"].alignments
                    values = []
                    for i in range(start, end):
                        cn = input_sheet[i, "CN"]
                        es = input_sheet[i, "ES"]
                        terms = input_sheet[i, "TERMS"]
                        terms_es = json.dumps(output_list[i-start], ensure_ascii=False)
                        values.append({ "CN": cn, "ES": es, "TERMS": terms, "TERMS_ES": terms_es })
                    chunks_iterator.complete_chunk((start, end), values)
                    total_attempts += state["attempts"]
                    print(f"========== Chunk ({start} - {end}) done ==========\n")

            # log token usage
            print("Cached tokens", cb.prompt_tokens_cached)
            print("Input tokens", cb.prompt_tokens)
            print("Output tokens", cb.completion_tokens) 
            total_cached_tokens += cb.prompt_tokens_cached
            total_input_tokens += cb.prompt_tokens
            total_output_tokens += cb.completion_tokens
        
        # logs
        print("Saving Alignment Sheet...")
        output_sheet.save()
        print("Alignment sheet saved to: ", OUTPUT_PATH)
        print("Total cached tokens so far:", total_cached_tokens)
        print("Total input tokens so far:", total_input_tokens)
        print("Total output tokens so far:", total_output_tokens)
        print("Total attempts so far:", total_attempts)
        print('#'*100)
        print("\n")

        if key_checker.has_key_pressed(f"{EXIT_HOT_KEY}"):
            print(f"Exit hotkeys was pressed. Exiting...")
            break