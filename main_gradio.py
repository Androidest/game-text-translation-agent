import gradio as gr
import pandas as pd
from pathlib import Path
from utils import *
from translation_agent.agent import create_translation_agent
from translation_agent import RAGChunkDispatcher
from p3_term_retrieval import TermRetriever
from p4_RAG import RAG
import pdb 

PATH_TEMP = PATH_PROJECT_ROOT/'data_temp'
PATH_TERMS = PATH_PROJECT_ROOT/'data_terms' 
PATH_RAG = PATH_PROJECT_ROOT/'data_rag' 

def scan_rag_files() -> list[str]:
    path = PATH_RAG
    file_extensions = [".xlsx", ".index"]
    
    names = set()
    if path.exists():
        for ext in file_extensions:
            for file_path in path.rglob(f"*{ext}"):
                names.add(file_path.stem)

    rag_files = []
    for name in names:
        if (path/f"{name}.index").exists() and (path/f"{name}.xlsx").exists():
            rag_files.append(name)
    
    rag_files = sorted(rag_files)
    return rag_files

def scan_terms_files() -> list[str]:
    path = PATH_TERMS        
    file_extensions = [".xlsx"]
    
    names = set()
    if path.exists():
        for ext in file_extensions:
            for file_path in path.rglob(f"*{ext}"):
                names.add(file_path.stem)
    
    terms_files = sorted(list(names))
    return terms_files

def on_load_sheet(file):
    if file is None:
        return pd.DataFrame()

    path  = PATH_TEMP/Path(file.name).parent.name/Path(file.name).name
    if path.exists():
        sheet = Sheet(path)
    else:
        sheet = Sheet(file.name)
        sheet.excel_file_path = path

        if sheet.dataframe.shape[1] < 4:
            sheet[:, "ES"] = ""
            sheet[:, "TERMS"] = ""
            sheet[:, "RAG"] = ""
        sheet.save()
    
    print(f"成功加载数据，形状为: {sheet.dataframe.shape}") 
    print(f"缓存文件：{file.name}") 
    print(f"临时文件：{sheet.excel_file_path}") 
    return sheet, sheet.dataframe 

def on_change_sheet(sheet:Sheet, dataframe:pd.DataFrame):
    if sheet is None:
        return
    
    sheet.dataframe = dataframe
    sheet.save()
    print(f"已保存：{sheet.excel_file_path}")

def refresh_chunked_agent(chunked_agent, llm:str, rag_filename:str, terms_filename:str, sheet:Sheet):
    index_path = PATH_RAG/f"{rag_filename}.index"
    text_sheet_path = PATH_RAG/f"{rag_filename}.xlsx"
    terms_path = PATH_TERMS/f"{terms_filename}.xlsx"

    agent = create_translation_agent(
        model=get_llm(llm), 
        max_attempts=5
    )

    if chunked_agent is not None and \
        chunked_agent.rag.index_path == index_path and \
        chunked_agent.rag.text_sheet_path == text_sheet_path and \
        chunked_agent.term_retriever.terms_path == terms_path and \
        chunked_agent.input_sheet.excel_file_path == sheet.excel_file_path:
        chunked_agent.agent = agent
        return chunked_agent
    
    rag = RAG(index_path=index_path, text_sheet_path=text_sheet_path)
    term_retriever = TermRetriever(term_alignment_path=terms_path)
    chunked_agent = RAGChunkDispatcher(
        chunk_size=4,
        parallel_chunks=20,
        input_sheet=sheet,
        output_sheet=sheet,
        agent=agent,
        desc="Translating",
        rag=rag,
        term_retriever=term_retriever,
    )
    return chunked_agent

async def on_translate(sheet:Sheet, chunked_agent, llm:str, rag_filename:str, terms_filename:str):
    def get_on_translate_return(progress=-1):
        running = progress == 0 or (chunked_agent is not None and chunked_agent.is_running)
        return (
            gr.File(interactive=not running),
            gr.Dataframe(value=sheet.dataframe, static_columns=[0,1,2,3] if running else [0,2,3], interactive=not running),
            gr.Button(f"翻译中... {progress*100:.2f}%" if running else "开始翻译", variant="secondary" if running else "primary"),
            gr.Button(visible=running),
            chunked_agent,
        )

    if sheet is None or rag_filename is None or terms_filename is None:
        yield get_on_translate_return()

    # 立即返回锁定状态
    print(f"开始翻译:{sheet.sheet_name}")
    yield get_on_translate_return(progress=0)
    
    try:
        chunked_agent = refresh_chunked_agent(chunked_agent, llm, rag_filename, terms_filename, sheet)
        async for progress in chunked_agent.async_run():
            yield get_on_translate_return(progress=progress)
        
        yield get_on_translate_return()
    
    except Exception as e:
        print(f"翻译过程中出错: {e}")
        chunked_agent.stop()
        yield get_on_translate_return()

def on_stop_translate(chunked_agent):
    if chunked_agent is not None and chunked_agent.is_running:
        chunked_agent.stop()

with gr.Blocks(title="Stephen 游戏文本翻译器") as demo:
        rag_names = scan_rag_files()
        terms_names = scan_terms_files()

        # region UI Components
        gr.Markdown("# Stephen 游戏文本翻译器")
        with gr.Row():
            with gr.Column():
                llm_dropdown = gr.Dropdown(choices=llm_names, label="选择LLM模型", interactive=True)
            with gr.Column():
                rag_dropdown = gr.Dropdown(choices=rag_names, label="选择RAG参考文件", interactive=True)
            with gr.Column():
                terms_dropdown = gr.Dropdown(choices=terms_names, label="选择术语库文件", interactive=True)
            with gr.Column():
                translate_btn = gr.Button("翻译", variant="secondary")
                stop_btn = gr.Button("停止翻译", variant="stop", visible=False)
        with gr.Row():
            file_input = gr.File(label="上传一个Excel文件（必须带'CN'列: 原始中文文本）", file_types=[".xlsx", ".xls"], file_count="single")
        dataframe_component = gr.Dataframe(
            label="Excel数据内容 (ES列可编辑)",
            type="pandas",
            interactive=True,  # 允许编辑
            col_count=(2, "fixed"),  # 设置列数，'dynamic'表示自适应。也可以指定固定数字，例如(3, 3)
            row_count=(0, "dynamic"),
            wrap=True,
            show_row_numbers=True,
            show_search=True,
            static_columns=[0,2,3],
            pinned_columns=1,
            column_widths=["30%", "65%", "30%", "40%"],
        )
        # endregion UI Components

        # region States
        llm_state = gr.State(value=llm_names[0] if len(llm_names) > 0 else None)
        rag_filename_state = gr.State(value=rag_names[0] if len(rag_names) > 0 else None)
        terms_filename_state = gr.State(value=terms_names[0] if len(terms_names) > 0 else None)
        sheet_state = gr.State(value=None)
        chunked_agent_state = gr.State(value=None)
        # endregion States

        # region Event Handlers
        llm_dropdown.change(
            fn=lambda x: x,
            inputs=[llm_dropdown],
            outputs=[llm_state],
        )
        
        rag_dropdown.change(
            fn=lambda x: x,
            inputs=[rag_dropdown],
            outputs=[rag_filename_state],
        )
        
        terms_dropdown.change(
            fn=lambda x: x,
            inputs=[terms_dropdown],
            outputs=[terms_filename_state],
        )

        file_input.change(
            fn=on_load_sheet, 
            inputs=[file_input],
            outputs=[sheet_state, dataframe_component]
        )

        dataframe_component.change(
            fn=on_change_sheet,
            inputs=[sheet_state, dataframe_component],
        )
        
        translate_btn.click(
            fn=on_translate,
            inputs=[sheet_state, chunked_agent_state, llm_state, rag_filename_state, terms_filename_state],
            outputs=[file_input, dataframe_component, translate_btn, stop_btn, chunked_agent_state]
        )

        stop_btn.click(
            fn=on_stop_translate,
            inputs=[chunked_agent_state]
        )
    
    # endregion Event Handlers

if __name__ == "__main__":
    demo.launch()