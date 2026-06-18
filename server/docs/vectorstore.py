# PDF -> CHUNKING 
#     ->MONGODB(FULL TEXT STORAGE) 
#     -> PINECONE (EMBEDDING+METADATA)

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.db import chunk_collection

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV","us-east-1")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","tutor-rags")

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY


UPLOAD_DIR="./upload_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

# pinecone global upsert fucntion

pc=None
index=None

def get_pinecone_index():
    global pc,index
    if index is None:
        pc=Pinecone(api_key=PINECONE_API_KEY)
        index=pc.Index(PINECONE_INDEX_NAME)
    return index



async  def load_vectorstore(uploaded_files,role:str,doc_id:str,grade:int):
    # initilize emedding model
    embed_model=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # get pinecone index
    pinecone_index=get_pinecone_index()
    # loop through uploaded files
    for file in uploaded_files:
        # 1. save raw file
        save_path=Path(UPLOAD_DIR) / file.filename
        with open(save_path,"wb") as f:
            f.write(file.file.read())
        # 2. load pdf text
        loader=PyPDFLoader(str(save_path))
        documents=loader.load()
        print(
            f"[RAG DEBUG][upload] file={file.filename!r} "
            f"pages_loaded={len(documents)}"
        )
        # 3. chunk text 
        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=splitter.split_documents(documents)
        print(
            f"[RAG DEBUG][upload] doc_id={doc_id} grade={grade!r} role={role!r} "
            f"chunks_created={len(chunks)}"
        )
        # 4. guard condition
        if not chunks:
            print(f"No text extracted from {file.filename}, skipping...")
            continue
        # 5. dual storing
        # 5.1 store full text in mongodb
        chunk_docs=[]
        for i,chunk in enumerate(chunks):
            chunk_docs.append({
                "chunk_id":f"{doc_id}-{i}",
                "doc_id":doc_id,
                "text":chunk.page_content,
                "page":int(chunk.metadata.get("page",0)),
                "source":file.filename,
                "grade":grade,
                "role":role,
            })
        if chunk_docs:
            insert_result=chunk_collection.insert_many(chunk_docs)
            print(
                f"[RAG DEBUG][upload] mongo_chunks_inserted="
                f"{len(insert_result.inserted_ids)} first_chunk_id={chunk_docs[0]['chunk_id']!r} "
                f"first_text_preview={chunk_docs[0]['text'][:120]!r}"
            )
        # 5.2 create embeddings
        texts=[chunk.page_content for chunk in chunks]
        embeddings= await asyncio.to_thread(embed_model.embed_documents,texts)
        print(
            f"[RAG DEBUG][upload] embeddings_created={len(embeddings)} "
            f"embedding_dim={len(embeddings[0]) if embeddings else 0}"
        )
        # upsert pinecone
        ids=[f"{doc_id}-{i}" for i in range(len(embeddings))]

        metadatas=[
            {
                "chunk_id":ids[i],
                "doc_id":doc_id,
                "page":int(chunks[i].metadata.get("page",0)),
                "source":file.filename,
                "grade":grade,
                "role":role,
            }
            for i in range(len(embeddings))
        ]

        vectors=list(zip(ids,embeddings,metadatas))
        print(
            f"[RAG DEBUG][upload] pinecone_upsert_count={len(vectors)} "
            f"first_vector_id={ids[0] if ids else None!r} "
            f"first_metadata={metadatas[0] if metadatas else None!r}"
        )
        upsert_response=await asyncio.to_thread(pinecone_index.upsert,vectors=vectors)
        print(f"[RAG DEBUG][upload] pinecone_upsert_response={upsert_response}")

    print(f"Successfully indexed {file.filename}")

