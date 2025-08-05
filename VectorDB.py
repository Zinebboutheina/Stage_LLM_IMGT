#Import & chargement des documents PMC
import pandas as pd
from langchain.schema import Document
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
#Embedding & création de la vector database 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
import os
import warnings
warnings.filterwarnings("ignore")

## chargement du fichier
logger.info("Loading File")
df = pd.read_parquet("Full_Articles4All_mAbs.parquet")
logger.success("Loading File OK")

# creation de la liste des document ainsi que des metadata
docs = []
logger.info("Creation of documents list")
for i, row in tqdm(df.iterrows()):
    text = row["fulltext"]
    pmc_id = row["pmc_id"]
    keyword= row["keyword"]
    doc = Document(page_content=text, metadata={"pmc_id": pmc_id, "antibody": keyword})
    docs.append(doc)

logger.success(f"Creation of documents list with length {len(docs)}")

## Chunk et ajout de chunk par article
#Découpage avec RecursiveCharacterTextSplitter et ajout des chunk_id
splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
split_docs = splitter.split_documents(docs)

logger.info("Add metadata in chunk")
# Ajout d'un ID unique pour chaque chunk pour traçabilité
for i, doc in tqdm(enumerate(split_docs)):
    doc.metadata["chunk_id"] = f"chunk_{i}"
logger.success("Add metadata in chunk ok")

## embedding model definition

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
logger.info("Adding document to vectorDB")
if os.path.isdir("ChromaDB"):
    logger.warning("Existed so don't create vectorDB !!!")
else: 
    logger.success(f"Does'nt Exist ==> creating ChromaDB")
    try: 
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory="ChromaDB"
        )
        vectorstore.persist()
        logger.success("Adding document to vectorDB OK")
    except Exception as e:
        logger.error(e)
