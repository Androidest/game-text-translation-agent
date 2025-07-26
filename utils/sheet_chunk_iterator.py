#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import Sheet
from tqdm import tqdm

class SheetParallelChunksIterator:
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


if __name__ == "__main__":
    s = Sheet("test.xlsx", default_data={ "CN": [], "ES": [] }, clear=True)
    l = [ { 'CN':f"cn{i}", "ES":f"es{i}" } for i in range(20)]
    l[2:7] = [None] * 5
    l[11:14] = [None] * 3
    for i in range(len(l)):
        s[i] = l[i]

    # s.save()
    print(s[:len(s)])

    s = Sheet("test.xlsx")
    it = SheetParallelChunksIterator(s, max_lines=27, chunk_size=2, chunks=2, desc="Processing chunks")
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