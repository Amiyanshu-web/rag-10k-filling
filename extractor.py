import os
import pathlib
from typing import List, Dict, Any

import orjson
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def extract_with_docling(pdf_path: str) -> Dict[str, Any]:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = False
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    conv_res = doc_converter.convert(pdf_path)
    
    text_content = []
    table_captions = []
    
    all_content = conv_res.document.export_to_dict()
    partial_text = all_content["texts"]
    
    for i in range(len(partial_text)):
        if partial_text[i]['label'] == 'caption' and (partial_text[i]['text'][:3].lower() in ['tab', 'table']):
            table_captions.append(partial_text[i]['text'])
        else:
            text_content.append(partial_text[i]['text'])
    
    text = "\n\n".join(text_content)
    
    head_list = []
    table_con = []
    
    for table_ix, table in enumerate(conv_res.document.tables):
        table_str = str(table)
        has_header = "column_header=True" in table_str
        head_list.append(1 if has_header else 0)
        
        table_df = table.export_to_dataframe()
        markdown_table = table_df.to_markdown()
        
        if has_header and table_ix < len(table_captions):
            table_con.append(markdown_table + "\n" + table_captions[table_ix])
        else:
            table_con.append(markdown_table)
    
    table_content = []
    leng = 0
    
    while leng < len(head_list):
        if head_list[leng] == 1:
            tab = table_con[leng]
            leng += 1
            
            if leng < len(head_list):
                while leng < len(head_list) and head_list[leng] == 0:
                    tab = tab + "\n" + table_con[leng]
                    leng += 1
            
            table_content.append(tab)
        else:
            table_content.append(table_con[leng])
            leng += 1
    
    return {"engine": "docling", "text": text, "tables": table_content}


def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec, option=orjson.OPT_NON_STR_KEYS))
            f.write(b"\n")


def run() -> List[str]:
    data_dir = "data"
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    pdf_paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, name))
    pdf_paths.sort()

    extracted_texts: List[str] = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting with Docling"):
        try:
            result = extract_with_docling(pdf_path)
            record = {
                "id": pathlib.Path(pdf_path).stem,
                "source_path": pdf_path,
                "engine": result.get("engine"),
                "text": result.get("text", ""),
                "tables": result.get("tables", []),
            }
            out_file = os.path.join(out_dir, f"{pathlib.Path(pdf_path).stem}.jsonl")
            write_jsonl([record], out_file)
            extracted_texts.append(record["text"]) 
        except Exception as e:
            record = {
                "id": pathlib.Path(pdf_path).stem,
                "source_path": pdf_path,
                "engine": "error",
                "error": str(e),
            }
            out_file = os.path.join(out_dir, f"{pathlib.Path(pdf_path).stem}.jsonl")
            write_jsonl([record], out_file)
            extracted_texts.append("")

    return extracted_texts