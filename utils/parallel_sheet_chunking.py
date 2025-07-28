#%%
from utils.sheet import Sheet
from utils.structured_validating_agent import StructuredValidatingState
from utils.key_strokle_listener import keyStrokeListener
from langgraph.graph import StateGraph
from tqdm import tqdm
import asyncio
import json
from langchain_community.callbacks.manager import get_openai_callback


class ParallelSheetChunksIterator:
    def __init__(self, sheet: Sheet, max_lines:int, chunk_size:int, chunks:int, desc:str):
        self.sheet = sheet
        self.max_lines = max_lines
        self.chunk_size = chunk_size
        self.chunks = chunks
        self.ongoing_chunks = set()

        i = 0
        proccessed_lines = len(sheet)
        while i < len(sheet):
            if sheet[i].isnull().all(axis=0):
                j = i + 1
                b_end = min(i + chunk_size, len(sheet))
                while j < b_end and sheet[j].isnull().all(axis=0):
                    j += 1
                
                self.ongoing_chunks.add((i, j))
                proccessed_lines -= j - i
                i = j
            else:
                i += 1

        self.progress = tqdm(
            total=max_lines,
            initial=proccessed_lines,
            desc=desc
        )
    
    def get_ongoing_chunks(self):
        remaining = self.chunks - len(self.ongoing_chunks) 
        if remaining > 0:
            start = len(self.sheet)
            end = min(start + remaining*self.chunk_size, self.max_lines)
            if end == start:
                return list(self.ongoing_chunks)

            indexes = list(range(start, end, self.chunk_size)) + [end]
            l = indexes[:-1]
            r = indexes[1:]
            new_chunks = [ (i,j) for i, j in zip(l, r) ]
            self.ongoing_chunks.update(new_chunks)

            for i in range(start, end):
                self.sheet[i] = None

        return list(self.ongoing_chunks)[:self.chunks]

    def complete_chunk(self, chunk_indexes:tuple, values:list):
        if chunk_indexes in self.ongoing_chunks:
            self.ongoing_chunks.remove(chunk_indexes)
            self.progress.update(chunk_indexes[1] - chunk_indexes[0])
            for i in range(chunk_indexes[0], chunk_indexes[1]):
                self.sheet[i] = values[i - chunk_indexes[0]]

    def __iter__(self):
        while True:
            chunks = self.get_ongoing_chunks()
            if len(chunks) == 0:
                self.progress.close()
                break
            yield chunks

class ParallelSheetChunkDispatcher:
    def __init__(
            self, 
            chunk_size:int, 
            parallel_chunks:int,
            input_sheet:Sheet,
            output_sheet:Sheet,
            agent:StateGraph,
            desc:str="",
            exit_hot_key:str="esc",
            ):
        
        self.chunk_size = chunk_size
        self.parallel_chunks = parallel_chunks
        self.input_sheet = input_sheet
        self.output_sheet = output_sheet
        self.agent = agent
        self.desc = desc
        self.key_checker = keyStrokeListener()
        self.key_checker.add_hotkey(exit_hot_key)
        self.exit_hot_key = exit_hot_key

    async def async_process_chunk(self, chunk_index:tuple) -> tuple[StructuredValidatingState, str]:
        
        state = self.input_chunk_to_agent(chunk_index)

        logs = ""
        async for step in self.agent.astream(state):
            node, state_update = next(iter(step.items())) 
            state.update(state_update)
            # if node == "generate_node":
            #     logs += f"Node: [{node}] output: {state_update.get("output_text")}, error:{state_update.get("error")}\n"
            if node == "validate_node":
                logs += f"\nNode: [{node}] attempt: {state.get('attempts', 0)}, error:{state_update.get("error")}"

        return state, logs

    async def async_process_multiple_chunks(self, chunk_list:list) -> list[tuple[StructuredValidatingState, str]]:
        tasks = [self.async_process_chunk(chunk_index) for chunk_index in chunk_list]
        return await asyncio.gather(*tasks)

    def run(self, on_update:callable = None):
        start = len(self.output_sheet)
        max_lines = len(self.input_sheet)
        total_cached_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_attempts = 0
        chunks_iterator = ParallelSheetChunksIterator(self.output_sheet, max_lines, self.chunk_size, self.parallel_chunks, desc=self.desc)
        
        for chunk_list in chunks_iterator:
            print('#'*100)
            print(f"Processing chunks: {json.dumps(chunk_list, ensure_ascii=False)}")
            print(f"Note: Press '{self.exit_hot_key}' to exit safely at any time.")

            with get_openai_callback() as cb:
                # process chunks in parallel
                results = asyncio.run(self.async_process_multiple_chunks(chunk_list))
                # process chunks results
                for (start, end), (state, logs) in zip(chunk_list, results):
                    print(f"========== Chunk ({start} - {end}) starts ==========")
                    print(f"\nbatch_str_len: {len(state["input_text"])}")
                    print(logs)

                    # validate a chunk
                    if state.get("error"):
                        print(f"Error from line ({start} - {end})", end="\n")
                        print("Input text:\n", state["input_text"])
                        print("Wrong output text:\n", state["output_text"])
                    # complete a chunk
                    else:
                        values = self.output_chunk_to_sheet((start, end), state)
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
            print("Saving output sheet...")
            self.output_sheet.save()
            print("Sheet saved to: ", self.output_sheet.excel_file_path)
            print("Total cached tokens so far:", total_cached_tokens)
            print("Total input tokens so far:", total_input_tokens)
            print("Total output tokens so far:", total_output_tokens)
            print("Total attempts so far:", total_attempts)
            print('#'*100)
            print("\n")

            if on_update:
                on_update(self)

            if self.is_pressed_exit():
                print(f"Exit hotkeys was pressed. Exiting...")
                break

    def is_pressed_exit(self):
        return self.key_checker.has_key_pressed(f"{self.exit_hot_key}")

    def input_chunk_to_agent(self, chunk_index:tuple) -> StructuredValidatingState:
        raise NotImplementedError("input_chunk_to_agent(self, chunk_index:tuple) -> StructuredValidatingState\nmust be implemented")
    
    def output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]:
        raise NotImplementedError("output_chunk_to_sheet(self, chunk_index:tuple, state:StructuredValidatingState) -> list[dict]\nmust be implemented")

if __name__ == "__main__":
    s = Sheet("test.xlsx", default_data={ "CN": [], "ES": [] }, clear=True)
    l = [ { 'CN':f"cn{i}", "ES":f"es{i}" } for i in range(20)]
    l[2:7] = [None] * 5
    l[11:14] = [None] * 3
    for i in range(len(l)):
        s[i] = l[i]

    s.save()
    print(s[:len(s)])

    s = Sheet("test.xlsx")
    it = ParallelSheetChunksIterator(s, max_lines=27, chunk_size=2, chunks=2, desc="Processing chunks")
    for i, parallel_chunks in enumerate(it):
        for chunk in parallel_chunks:
            print("Processing chunk:", chunk)
            values = [ {'CN':f"cn{k}", "ES":f"es{k}" } for k in range(chunk[0], chunk[1]) ]
            it.complete_chunk(chunk, values)
            print("Completed chunk:", chunk)
            print("-"*20)
        print("#"*50)
        if i == 1:
            break

    # s.save()
    print(s[:len(s)])