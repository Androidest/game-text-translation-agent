#%%
import pandas as pd
import os

class Sheet:
    def __init__(self, excel_file_path:str, sheet_name:str=None, default_data:dict=None, clear:bool=False):
        self.excel_file_path = excel_file_path
        self.sheet_name = sheet_name
        if default_data is not None and (clear or not os.path.exists(excel_file_path)):
            if sheet_name is None:
                self.sheet_name = 'Sheet1'
            self.dataframe = pd.DataFrame(default_data)
        else:
            try:
                with pd.ExcelFile(excel_file_path) as excel_file:
                    sheet_names = excel_file.sheet_names
                    for name in sheet_names:
                        if sheet_name is not None and name != sheet_name:
                            continue
                        df = excel_file.parse(name)
                        self.dataframe = df
                        self.sheet_name = name
                        break
            except FileNotFoundError:
                raise FileNotFoundError(f"File '{excel_file_path}' not found")
            except Exception as e:
                raise Exception(f"Error reading file '{excel_file_path}': {e}")
    
    def __len__(self):
        return self.dataframe.shape[0]

    def __iter__(self):
        for i in range(self.dataframe.shape[0]):
            yield self.dataframe.values[i]
    
    def __getitem__(self, indices):
        return self.dataframe.loc[indices]
    
    def __setitem__(self, indices, value):
        self.dataframe.loc[indices] = value

    def append(self, value):
        self[len(self)] = value

    def column_names(self):
        return self.dataframe.columns
    
    def save(self):
        dirname = os.path.dirname(os.path.abspath(self.excel_file_path))
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dataframe.to_excel(self.excel_file_path, sheet_name=self.sheet_name, index=False)

if __name__ == "__main__":
    # read exsisting excel file
    data_sheet = Sheet('TLON语言包ES 20250701.xlsx')
    print(data_sheet.column_names())
    print(data_sheet[0, "CN"])
    print(data_sheet[5])
    for i, (row1, row2) in enumerate(data_sheet):
        print(row1, row2)
        if i == 10:
            break
    
    # create new excel file
    data_sheet = Sheet('b.xlsx', default_data={'A':[], 'B':[], 'C':[], 'D':[]})
    data_sheet[0, "C"] = 1123
    data_sheet[3] = {'A':1, 'D':2}
    data_sheet.save()