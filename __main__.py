import os
from utils import default_llm, get_llm, llm_names
from argparse import ArgumentParser
from translation_agent import translate

# parse args
parser = ArgumentParser()
parser.add_argument("--llm", type=str, help='choose an llm', default="")
args = parser.parse_args()
model = get_llm(args.llm)

# Select a model when --llm is not given
if model is None:
    while True:
        print("Supported AI models:")
        for i, name in enumerate(llm_names):
            print(f" - {i}. {name}")
        
        index = input("Please select a model by number:")
        try:
            model = get_llm(llm_names[int(index)])
            break
        except:
            print("Wrong input!")

print(f"Using model: {model.model_name}")

def is_excel_file(file_path):
    excel_extensions = ('.xlsx', '.xls', '.xlsm', '.xlsb', '.xltx', '.xltm')
    return file_path.lower().endswith(excel_extensions)

def on_update(dispatcher):
    pass

while True:
    print("------------- Stephen Game Text Translator -------------")
    file_path = input("Please enter the file path: ")
    # Validate if the path is empty
    if not file_path:
        print("File path cannot be empty!")
        continue
    elif not os.path.exists(file_path):
        print("File does not exist!")
        continue
    elif not os.path.isfile(file_path) or not is_excel_file(file_path):
        print(f"The file '{file_path}' you provided is not an Excel file!")
        continue
    else:
        print(f"You want to translate the file: {file_path}")

    translate(model, file_path, on_update=on_update)
    print(f"Translation is complete! File has been saved to: {file_path}\n\n")