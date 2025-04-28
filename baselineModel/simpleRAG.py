""" 
This script implements a simple RAG (Retrieval-Augmented Generation) system using Azure OpenAI and FAISS for document retrieval.
It loads data from CSV files, creates documents, generates embeddings, builds a FAISS index, and retrieves relevant documents based on user queries.
It also generates answers using the Azure OpenAI model based on the retrieved documents.
The script is designed to be run in an environment with access to Azure services and requires the installation of several libraries.

"""
import os
import json
import asyncio
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from types import SimpleNamespace
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------
# 1. Load Configuration
# -----------------------------------
def load_config():
    config_path = r"config.json"
    with open(config_path) as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))

config = load_config()

# -----------------------------------
# 2. Initialize Azure Client (for Chat/Completion)
# -----------------------------------
def initialize_azure_client(config):
    client = SecretClient(vault_url=config.key_vault_url, credential=DefaultAzureCredential())
    secret = client.get_secret(config.dev_secret_name)
    return AzureOpenAI(
        api_key=secret.value,
        api_version=config.chat.api_version,
        azure_endpoint=config.chat.azure_endpoint
    )

azure_client = initialize_azure_client(config)

# -----------------------------------
# 3. Load CSV Files
# -----------------------------------
app_file = r"data/application_master.csv"   # Application data
app_emp_file = r"data/apps_owners.csv"       # Application connecting with employees
emp_file = r"data/Fin_Emp.csv"               # Employee data
org_file = r"data/Fin_Org.csv"               # Organisation connecting with employees
proc_file = r"data/process_master.csv"     # Process data
proc_org_file = r"data/orgs_processes.csv" # Process connecting with organisation
proc_app_file = r"data/process_applications.csv"  # Processes connecting applications
proc_emp_file = r"data/process_owners.csv"       # Process connecting with employees

print("Loading CSV files...")
app_df = pd.read_csv(app_file)
app_emp_df = pd.read_csv(app_emp_file)
emp_df = pd.read_csv(emp_file, low_memory=False)
org_df = pd.read_csv(org_file)
proc_df = pd.read_csv(proc_file)
proc_org_df = pd.read_csv(proc_org_file)
proc_app_df = pd.read_csv(proc_app_file)
proc_emp_df = pd.read_csv(proc_emp_file)

# -----------------------------------
# 4. Create Documents
# -----------------------------------
def create_document(entity_type, row, extra_info=""):
    fields = [f"{col}: {row[col]}" for col in row.index if pd.notnull(row[col])]
    doc_text = f"{entity_type.upper()} DATA: " + " | ".join(fields)
    if extra_info:
        doc_text += " | " + extra_info
    metadata = {"entity_type": entity_type, "id": row.get("id", None)}
    return {"doc_id": f"{entity_type}_{row.get('id', '')}", "text": doc_text, "metadata": metadata}

print("Creating documents...")
documents = []

# Application documents
for _, row in app_df.iterrows():
    linked_emp = app_emp_df[app_emp_df['app_id'] == row['id']]
    emp_info = ", ".join(f"employee_id: {lrow['employee_id']} (is_owner: {lrow['is_owners']})" 
                        for _, lrow in linked_emp.iterrows())
    extra = f"Linked employees: {emp_info} | App Org: {row.get('app_org', '')}"
    documents.append(create_document("application", row, extra))

# Employee documents
for _, row in emp_df.iterrows():
    extra = f"Org ID: {row.get('org_id', '')}, Line Manager ID: {row.get('line_manager_id', '')}"
    documents.append(create_document("employee", row, extra))

# Organisation documents
for _, row in org_df.iterrows():
    extra = f"Org Head: {row.get('org_head', '')}, Parent Org ID: {row.get('parent_org_id', '')}"
    documents.append(create_document("organisation", row, extra))

# Process documents
for _, row in proc_df.iterrows():
    linked_org = proc_org_df[proc_org_df['process_id'] == row['id']]
    org_ids = ", ".join(f"org_id: {r['org_id']}" for _, r in linked_org.iterrows())
    linked_app = proc_app_df[proc_app_df['process_id'] == row['id']]
    app_ids = ", ".join(f"application_id: {r['application_id']}" for _, r in linked_app.iterrows())
    linked_emp = proc_emp_df[proc_emp_df['process_id'] == row['id']]
    emp_ids = ", ".join(f"employee_id: {r['employee_id']} (is_owner: {r['is_owners']})" 
                        for _, r in linked_emp.iterrows())
    extra = f"Linked Organisations: {org_ids} | Linked Applications: {app_ids} | Linked Employees: {emp_ids}"
    documents.append(create_document("process", row, extra))

print(f"Created {len(documents)} documents.")

# -----------------------------------
# 5. Embedding with Checkpointing and FAISS Index Building
# -----------------------------------
async def embed_batch_async(batch, embeddings):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embeddings.embed_documents, batch)

async def encode_documents_with_checkpoints(docs, embeddings, batch_size=32, checkpoint_interval=4000, checkpoint_dir="checkpoints", resume=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    existing_checkpoints = sorted(
        Path(checkpoint_dir).glob("embeddings_batch_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    embeddings_list = []
    start_idx = 0
    if resume and existing_checkpoints:
        total_embedded = 0
        for cp in existing_checkpoints:
            emb = np.load(cp)
            total_embedded += emb.shape[0]
        if total_embedded > len(docs):
            print("Checkpoint files are inconsistent with the current document count. Clearing checkpoints and starting fresh.")
            for cp in existing_checkpoints:
                os.remove(cp)
            start_idx = 0
        else:
            embeddings_list = [np.load(cp) for cp in existing_checkpoints]
            embeddings_combined = np.vstack(embeddings_list)
            start_idx = embeddings_combined.shape[0]
            embeddings_list = [embeddings_combined]
            print(f"Resuming from checkpoint at document {start_idx}")
    for idx in range(start_idx, len(docs), batch_size):
        batch_texts = [doc["text"] for doc in docs[idx: idx + batch_size]]
        embeddings_batch = await embed_batch_async(batch_texts, embeddings)
        embeddings_list.append(embeddings_batch)
        current_idx = idx + len(batch_texts)
        if current_idx % checkpoint_interval == 0 or current_idx >= len(docs):
            np.save(f"{checkpoint_dir}/embeddings_batch_{current_idx}.npy", np.vstack(embeddings_list))
            print(f"Checkpoint saved at {current_idx} documents.")
    return np.vstack(embeddings_list).astype('float32')

# Initialize embeddings (retrieve Azure API key)
client = SecretClient(vault_url=config.key_vault_url, credential=DefaultAzureCredential())
secret = client.get_secret(config.dev_secret_name)
azure_api_key = secret.value

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_api_key,
    azure_endpoint=config.chat.azure_endpoint,
    api_version=config.chat.api_version
)

async def build_index():
    embeddings_matrix = await encode_documents_with_checkpoints(documents, embeddings)
    faiss.normalize_L2(embeddings_matrix)
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)
    print(f"FAISS index built with {index.ntotal} documents.")
    return index

# Global vector store
vector_store = SimpleNamespace(index=None)

# -----------------------------------
# 6. Retrieval and RAG Implementation
# -----------------------------------
def retrieve(query, top_k=5):
    if vector_store.index is None:
        raise ValueError("FAISS index is not available for retrieval.")
    query_emb = np.array([embeddings.embed_query(query)]).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = vector_store.index.search(query_emb, top_k)
    results = [documents[idx] for idx in indices[0]]
    return results

def generate_answer_with_llm(query: str, top_documents):
    """
    Use Azure OpenAI to generate a final answer from the top retrieved documents.
    """
    config = load_config()
    llm = initialize_azure_client(config)
    context = "\n\n".join(top_documents)
    prompt = [
        {
            "role": "system",
            "content": f"""
You are an AI assistant tasked with answering a query based on the provided context about employees and organizations.
Please provide a detailed and well-structured answer to the user's question.
- Organize the answer into bullet points if appropriate.
- Use headings where relevant.
- Include all relevant details concisely.
Context:
{context}
Question: "{query}"
Provide a well-structured answer.
            """
        }
    ]
    response = llm.chat.completions.create(model=config.chat.model, messages=prompt)
    response_content = response.choices[0].message.content.strip()
    return response_content

def simple_rag(query, top_k=5):
    retrieved_docs = retrieve(query, top_k=top_k)
    top_docs_text = [doc["text"] for doc in retrieved_docs]
    print("top_docs_text:", top_docs_text)  # Debug print
    final_answer = generate_answer_with_llm(query, top_docs_text)
    return final_answer
def process_queries_from_excel():
    import time  
    query_excel_file = r"data/LLMEval_1.xlsx"      
    output_excel_file = r"Outputs/LLM_responses_simpleRAG_multihop.xlsx"
    queries_df = pd.read_excel(query_excel_file)
    # List of k values to test.
    k_values = [1, 2, 3, 5, 8, 13, 15, 21]
    for idx, row in queries_df.iterrows():
        query = row["Query"]
        for k in k_values:
            col_docs = f"k_{k}_docs"
            col_time = f"k_{k}_retrieve_time"
            col_responsetime = f"k_{k}_response_answer"
            col_answer = f"k_{k}_final_answer"
            start_time = time.time()
            retrieved_docs = retrieve(query, top_k=k)
            elapsed_time = time.time() - start_time
            # Concatenate the retrieved documents' text.
            docs_str = "\n\n----\n\n".join([doc["text"] for doc in retrieved_docs])
            top_docs_text = [doc["text"] for doc in retrieved_docs]
            start_time = time.time()
            final_answer = generate_answer_with_llm(query, top_docs_text)
            response_time = time.time() - start_time
            queries_df.at[idx, col_docs] = docs_str
            queries_df.at[idx, col_time] = elapsed_time
            queries_df.at[idx, col_responsetime] = response_time
            queries_df.at[idx, col_answer] = final_answer
            print(f"Processed query '{query[:50]}...' for k={k} in {elapsed_time:.2f} seconds.")
    queries_df.to_excel(output_excel_file, index=False)
    print(f"Results saved to {output_excel_file}")
# -----------------------------------
# 7. Main function
# -----------------------------------
async def main():
    global vector_store
    INDEX_FILE = "faiss_index.index"
    if os.path.exists(INDEX_FILE):
        print("FAISS index file found. Loading index from file...")
        index = faiss.read_index(INDEX_FILE)
    else:
        print("FAISS index file not found. Building FAISS index...")
        index = await build_index()
        print("Saving FAISS index to file...")
        faiss.write_index(index, INDEX_FILE)
    vector_store.index = index
    print("FAISS index is ready for retrieval.")
    print("vector_store.index:", vector_store.index) 
   
    process_queries_from_excel()

if __name__ == '__main__':
    asyncio.run(main())