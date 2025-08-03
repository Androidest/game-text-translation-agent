from P4_RAG.model import TextEmbModel
from utils import PATH_DATA, Sheet
from tqdm import tqdm

INPUT_SHEET_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"
OUTPUT_SHEET_PATH = PATH_DATA / "rag_text.xlsx"

print("Loading sheets...")
input_sheet = Sheet(INPUT_SHEET_PATH)
output_sheet = Sheet(OUTPUT_SHEET_PATH, conlumns=input_sheet.column_names(), clear=True)
model = TextEmbModel()

for i in tqdm(range(0, len(input_sheet)), desc="Cleaning"):
    text = input_sheet[i, "CN"]
    tokens = model.count_tokens(text)
    if tokens > model.max_length:
        continue
    output_sheet.append(input_sheet[i])

output_sheet.save()
print("Done!")




