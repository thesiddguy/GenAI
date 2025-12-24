from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from ocr_gemini import GeminiOCR
import chromadb
import re
import fitz
import pandas as pd
from docx import Document
import sqlite3

app = FastAPI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
gemini = GeminiOCR()

try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection("documents")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

file_store = {}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, filename="", page_num=0):
    if not text or len(text.strip()) < 50:
        return []
    
    text = text.strip()
    paragraphs = re.split(r'\n\s*\n+', text)
    chunks = []
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 50:
            continue
        
        if len(para) <= chunk_size:
            chunks.append({
                "text": para,
                "filename": filename,
                "page": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "filename": filename,
                            "page": page_num,
                            "chunk_index": chunk_index
                        })
                        chunk_index += 1
                    
                    words = sentence.split()
                    overlap_words = current_chunk.split()[-overlap//10:] if current_chunk else []
                    current_chunk = " ".join(overlap_words + words) + " "
            
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "filename": filename,
                    "page": page_num,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
    
    return chunks

def detect_file_type_in_question(question):
    question_lower = question.lower()
    
    if 'pdf' in question_lower:
        return 'pdf'
    elif 'txt' in question_lower or 'text' in question_lower:
        return 'txt'
    elif 'docx' in question_lower or 'docs' in question_lower or 'document' in question_lower:
        return 'docx'
    elif 'image' in question_lower or 'jpg' in question_lower or 'png' in question_lower or 'bmp' in question_lower or 'jpeg' in question_lower:
        return 'image'
    elif 'db' in question_lower or 'database' in question_lower:
        return 'db'
    elif 'csv' in question_lower:
        return 'csv'
    
    return None

@app.get("/")
def root():
    with open("static/index.html", "r", encoding='utf8') as f:
        return HTMLResponse(content=f.read())

@app.get("/query")
def query_page():
    with open("static/query.html", "r", encoding='utf8') as f:
        return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    file_ids = []
    file_info = []
    os.makedirs("uploads", exist_ok=True)

    for f in files:
        file_id = str(uuid.uuid4())[:8]
        path = os.path.join("uploads", f"{file_id}_{f.filename}")

        with open(path, "wb") as out:
            out.write(await f.read())

        ext = f.filename.lower().split('.')[-1] if '.' in f.filename else 'file'
        
        extracted_text = ""
        all_chunks = []
        
        try:
            if ext == 'pdf':
                doc = fitz.open(path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page_text = doc[page_num].get_text()
                    full_text += page_text + "\n\n\t\t"
                    page_chunks = create_chunks(page_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, page_num + 1)
                    all_chunks.extend(page_chunks)
                doc.close()
                extracted_text = full_text
            
            elif ext == 'docx':
                doc = Document(path)
                doc_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                doc_chunks = create_chunks(doc_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, 1)
                all_chunks.extend(doc_chunks)
                extracted_text = doc_text
            
            elif ext == 'txt':
                with open(path, 'r', encoding='utf-8') as txt_file:
                    txt_text = txt_file.read()
                txt_chunks = create_chunks(txt_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, 1)
                all_chunks.extend(txt_chunks)
                extracted_text = txt_text
            
            elif ext in ['jpg', 'jpeg', 'png', 'bmp']:
                ocr_text = gemini.extract_text(path, "English")
                img_chunks = create_chunks(ocr_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, 1)
                all_chunks.extend(img_chunks)
                extracted_text = ocr_text
            
            elif ext == 'csv':
                df = pd.read_csv(path)
                csv_text = df.to_string()
                csv_chunks = create_chunks(csv_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, 1)
                all_chunks.extend(csv_chunks)
                extracted_text = csv_text
            
            elif ext == 'db':
                conn = sqlite3.connect(path)
                db_text = pd.read_sql_query("SELECT * FROM data_table;", conn).to_string()
                db_chunks = create_chunks(db_text, CHUNK_SIZE, CHUNK_OVERLAP, f.filename, 1)
                all_chunks.extend(db_chunks)
                conn.close()
                extracted_text = db_text
        
        except Exception as e:
            extracted_text = f"Error extracting text: {str(e)}"
            all_chunks = [{"text": extracted_text, "filename": f.filename, "page": 1, "chunk_index": 0}]

        file_store[file_id] = {
            "filename": f.filename,
            "path": path,
            "type": ext,
            "full_text": extracted_text,
            "extracted_text": "\n\n".join([chunk["text"] for chunk in all_chunks])
        }
        
        for i, chunk in enumerate(all_chunks):
            if chunk["text"].strip() and len(chunk["text"]) > 50:
                embedding = gemini.get_embeddings(chunk["text"])
                if embedding:
                    doc_id = f"{file_id}_chunk_{i}"
                    metadata = {
                        "file_id": file_id,
                        "filename": chunk["filename"],
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": len(all_chunks),
                        "file_type": ext
                    }
                    collection.add(
                        documents=[chunk["text"]],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
        
        file_info.append({
            "id": file_id,
            "name": f.filename,
            "type": ext,
            "text_preview": all_chunks[0]["text"][:200] + "..." if all_chunks else "",
            "chunk_count": len(all_chunks)
        })

        file_ids.append(file_id)

    return {"file_ids": file_ids, "files": file_info}

@app.post("/api/query")
async def query_llm(
    question=Form(""),
    file_ids=Form("[]"),
    image: UploadFile = File(None)
):
    full_question = question.strip()
    
    detected_file_type = detect_file_type_in_question(full_question)
    
    is_generic = detected_file_type is not None
    
    context = ""
    source = ""
    
    if is_generic and detected_file_type:
        target_file_id = None
        
        for fid, fdata in file_store.items():
            file_ext = fdata["type"]
            
            if detected_file_type == 'image' and file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
                target_file_id = fid
                break
            elif detected_file_type == file_ext:
                target_file_id = fid
                break
        
        if target_file_id:
            context = file_store[target_file_id].get("full_text", "")
            source = file_store[target_file_id]["filename"]
        
        if context:
            prompt = f"""You are a helpful assistant. Describe the document content briefly.

            Document Content:
            {context}

            Question: 
            {question}

            Instructions:
            - Provide a brief and comprehensive summary
            - Be clear and informative

            Answer:"""
        else:
            return {
                "context": "",
                "answer": "No file of this type found in uploaded documents.",
                "source": ""
            }
    else:
        question_embedding = gemini.get_embeddings(full_question)
        if not question_embedding:
            return {"context": "", "answer": "Error generating embeddings.", "source": ""}

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if results["documents"] and results["documents"][0]:
            max_distance = -1
            best_document = ""
            best_metadata = None

            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                if distance > max_distance:
                    max_distance = distance
                    best_document = doc
                    best_metadata = results["metadatas"][0][i]

            context = best_document
            metadata = best_metadata

            filename = metadata.get("filename", "")
            ext = os.path.splitext(filename)[1].lower()
            page = metadata.get("page", 1)

            if ext in [".pdf", ".docx"]:
                source = f"page {page} of {filename}"
            else:
                source = filename
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.

        Context:
        {context}

        Question: 
        {question}

        Instructions:
        - Use ONLY information from the Context above
        - Answer the question
        - If the answer is NOT in the context, respond exactly: "I couldn't find this information in the provided documents."

        Answer:"""
    gemini_response = gemini.answer(prompt, "English")
    
    return {
        "context": context,
        "answer": gemini_response,
        "source": source
    }
