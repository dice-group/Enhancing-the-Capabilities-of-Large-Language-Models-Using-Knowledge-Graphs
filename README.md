# Subgraph Retrieval Augmented Generation over RDF Knowledge Graphs

This repository is originally named as **Enhancing the Capabilities of Large Language Models Using Knowledge Graphs in Finance**.
---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Data & Knowledge Graph](#data--knowledge-graph)  
4. [Prerequisites](#prerequisites)  
5. [Installation & Setup](#installation--setup)  
6. [Usage](#usage)  
7. [Python Scripts Overview](#python--scripts--overview)  
8. [Generating Requirements File](#generating--requirements--file)  
9. [Acknowledgments](#acknowledgments)   

---

## Overview

This project investigates how integrating **Knowledge Graphs (KGs)** with **Large Language Models (LLMs)** can improve accuracy and reasoning in organizational-domain question answering. We evaluate:

- **Traditional Retrieval** (TF-IDF, BM25) vs. **Dense RAG** (DPR)  
- **KG-Enhanced Retrieval** (subgraph vs. triple chunking + fusion)  
- **SPARQL-Driven Querying** (LLM→SPARQL→KG)  

---

## Data & Knowledge Graph

### Raw Data

- **Source**: Eight CSV/Excel tables from four dimentions(employees, organizations, applications, processes).  
- **Format**: original, un-anonymized tables—preserve full fidelity and FK relationships.  
- **Location**: stored off-repo (internal).

### Knowledge Graph

- **Construction**: `kgCreation/CompleteFINKG.ipynb` uses **rdflib** + `ontology_schema.jsonld`.  
- **Output**: `kgCreation/ExtendedFinKG_anonymized.ttl` (Turtle).  
- **Anonymization**: Node IDs & predicates are pseudonymized; The overall structure (the types of entities and how they relate) still uses clear, readable names.

### Ground Truth

- **Anonymized Q&A**: `anonymize/groundTruth_anonymized.xlsx` contains expert-verified queries & answers.  
- **Regeneration**: run `anonymize/anonymize.ipynb` to adjust or re-anonymize.

---

## Prerequisites

- **Python 3.8+**  
- **Conda** (recommended) or `virtualenv` + `pip`  
- **Git**  
- _(Optional)_ **Java 8+** for Blazegraph

---

## Installation & Setup

### 1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/anubhuti_master_thesis-1-main.git
   cd anubhuti_master_thesis-1-main
   ```

### 2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```

### 3. **Install required Python packages** (if a `requirements.txt` is already present):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Install Blazegraph on Windows
1. **Download Blazegraph**  
   - Visit the [Blazegraph Releases](https://github.com/blazegraph/database/releases) page.
   - Download the `blazegraph.jar` file.

2. **Run Blazegraph**  
   1. Place the `blazegraph.jar` file in a folder (e.g., `C:\Blazegraph`).
   2. Open Command Prompt and navigate to that folder:
      ```bash
      cd C:\Blazegraph
      ```
   3. Start Blazegraph:
      ```bash
      java -server -Xmx4g -jar blazegraph.jar
      ```
   4. Once Blazegraph is running, you can access it at:
      
    http://localhost:9999/blazegraph
      
   5. Navigate to "Namespace" in the UI and create a new namespace with:
    - Name: `myGraph`
    - Mode: `triples`
    Default namespace `kb`

    Update the namespace in the blazegraph url in all the files using graphDB `blazegraph`.
    http://localhost:9999/blazegraph/namespace/myGraph/sparql
    
## Python Scripts Overview
Below is an overview of the key Python files and their functionalities:

### **1. ` baselineModel/`**  
Implements baseline retrieval approaches.
1. `BM25.ipynb` – Lexical retrieval with BM25.
2. `DPR.ipynb` – Dense retrieval using transformers.
3. `tf-idf.ipynb` – TF-IDF based baseline.
4. `simpleRAG.py` – A simple RAG (Retrieval-Augmented Generation) prototype using CSV data.

### **2. `blazegraph/`**  
  SPARQL query generation and evaluation using Blazegraph.
1. `eval.ipynb` – Approach evaluation using metrics (Precision, Recall, F1, MRR, hits@k,Rougue, Bleu, chrF).
2. `LLMApache.ipynb` – Query generation and validation with LLM and using Apache Jena Fuseki graph database.
3. `LLMBlazegraphVal.ipynb` – Batch SPARQL generation and LLM answer generation from Excel queries using Blazegraph graph database.

### **3. `kgCreation/`**  
  Knowledge graph construction and schema management.
1. `CompleteFINKG.ipynb` – Builds the complete financial knowledge graph from CSVs and RDF triples.
2. `ExtendedFinKG_anonymized.ttl` – An anonymized RDF Turtle file representing an extended version of the financial knowledge graph.
3. `ontology_schema.jsonld` – JSON-LD representation of the ontology used to define schema and classes in the knowledge graph.

### **4. `LangGraph/`**  
  LangGraph-based RAG pipelines.
1. `LG_hybrid_subgraph.ipynb` – Retrieves and processes hybrid subgraphs using LLM-assisted query parsing and subgraph ranking.
2. `LG_hybrid_triple.ipynb` – Focuses on triple-level retrieval and evaluation within the LangGraph pipeline.
3. `LG_LLMblazegraph.ipynb` – Combines LangGraph and Blazegraph to perform end-to-end retrieval and answer generation via SPARQL.

### **5. `RDF-RAG/`**  
  Final and hybridized implementation of the RDF-RAG pipeline.
1. `hybrid.ipynb` - Main hybrid retrieval pipeline that uses both lexical and dense retrieval (via RRF), LLM parsing, subgraph linking, and natural language generation.
2. `hybrid_triple.ipynb` - Variant of the hybrid model working at the triple-level rather than the subgraph level.

## Generating Requirements File
To automatically generate a `requirements.txt` file based on the libraries used in the project, you can use any of the following methods:

   Make sure all your dependencies are installed in your current Python environment, then run:
   ```bash
   pip freeze > requirements.txt
   ```
   This will create (or overwrite) a `requirements.txt` file listing the exact versions of the installed packages.
 
## **Acknowledgments**
- Siemens Energy for providing computational resources and domain expertise.
- University of Paderborn for academic support and guidance.
 
## Contact
This work is mostly done by Anubhuti Singh during her **Master’s Thesis** submitted April 2025  

For any questions or wishes, please contact:
Anubhuti Singh at anubhuti@mail.uni-paderborn.de.
 
 
