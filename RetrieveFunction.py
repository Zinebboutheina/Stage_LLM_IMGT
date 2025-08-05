
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA, StuffDocumentsChain #une chaîne plus simple qui "concatène" les documents et applique un LLM dessus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate #Sert à créer des prompts dynamiques, avec des variables comme {context}
from langchain.schema.document import Document
from collections import defaultdict #Pour regrouper automatiquement les documents par pmc_id dans des listes.
from uuid import uuid4
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings("ignore")


#  Embedding model pour requêtes (doit être identique à celui de la DB)
embedding_model = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

#  Chargement de la base vectorielle Chroma existante
vectorstore = Chroma(
    persist_directory="./ChromaDB/",
    embedding_function=embedding_model
)

#  Chargement du LLM
llm = OllamaLLM(model="phi4")




def get_mab_info_from_chroma_no_temp(mab_name, vectorstore, llm, k=10): #Fonction principale, elle prend :le nom du mAb; vectorstore : la base ChromaDB contenant les chunks; llm utilisé
    logger.info(f"🔎 Extraction pour : {mab_name}")

    # Étape 1 : récupération de tous les documents du mAb
    all_docs = vectorstore.similarity_search(mab_name, k=k, filter={"antibody": mab_name})
    logger.info(f"{len(all_docs)} chunks trouvés pour {mab_name}") #Rechercher les  chunks les plus proches du nom du mAb, filtre pour que ce soit bien les chunks annotés pour ce mAb seulement.

    # Étape 2 : regroupement par pmc_id
    pmc_chunks = defaultdict(list)
    for doc in all_docs:
        pmc_chunks[doc.metadata["pmc_id"]].append(doc) #je regroupe les chunks par pmc_id pour traiter chaque article séparément.

    # Prompts avec {context}
    target_question = (
        "Based only on the content of this scientific article (including the abstract and main body), "
        "what is the biological target of the monoclonal antibody?\n\n"
        "The answer can be the official target name (e.g., PD-L1, EGFR) or any known synonym or full biological name, "
        "expressed in one or several words.\n"
        "If the article mentions multiple possible targets or pathways, include all relevant ones.\n"
        "If the target is not mentioned or cannot be inferred, simply answer: Not found.\n\n{context}"
    )

    domain_prompt = (
        "You are a biomedical classification expert. \n"
        "Identify the main biomedical research domain of the article below.\n"
        "Return ONLY one or two words, all lowercase, no punctuation.\n\n"
        "Examples: oncology, immunology, virology, etc.\n\n{context}"
    )

    effect_q = (
        f"You are a biomedical expert. Extract the main **therapeutic effect** of the monoclonal antibody '{mab_name}' from the article.\n"
        "Return a short phrase or sentence. No explanation.\n\n{context}"
    )

    mech_q = (
        f"You are a biomedical expert. Describe the **mechanism of action** of '{mab_name}' as a short phrase.\n\n{{context}}"
    )

    response = (
        f"You are a biomedical expert. Based on the content of this article about '{mab_name}', "
        "write a concise description (max 5 sentences) covering its target, usage, mechanism and therapeutic effect.\n\n{context}"
    )

    global_prompt = (
        "You are a biomedical expert.\n\n"
        "You are provided with several short descriptions of a monoclonal antibody. "
        "Write a global synthesis highlighting:\n"
        "- most common mechanisms of action\n- biological targets\n- therapeutic areas.\n"
        "One paragraph only.\n\n{context}"
    )

    results, descriptions, domains, effects, mechanisms = [], [], [], [], [] #Cinq listes pour stocker les résultats extraits de chaque type (target, description, etc.)

    for pmc_id, docs in pmc_chunks.items(): #je traite chaque article (pmc_id) individuellement
        try:
            # Séparation des retrievers pour chaque tâche
            # On les restreint aux `docs` uniquement pour simuler une recherche dans un seul article
            retriever_target = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"pmc_id": pmc_id}}),
                chain_type="stuff"
            )
            target_chain = StuffDocumentsChain(  #Chaîne LLM spéciale qui : Prend plusieurs documents, Les concatène en un seul bloc de texte (stuffing), Remplace {context} dans le prompt par ce bloc, Passe le tout au LLM pour générer une réponse.
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(target_question), output_key="result"), #transforme le texte du prompt (avec {context}) en un template utilisable
                       #output_key="result" : indique que la réponse du LLM sera dans la variable result
                document_variable_name="context"                   # dit que les documents vont être insérés dans {context},c’est cette chaîne qui formule intelligemment la question à partir des bons documents.
            )
            target_chunks = vectorstore.similarity_search(target_question, k=3, filter={"pmc_id": pmc_id}) #recherche des chunks les plus pertinents : en utilisant le prompt de la question comme requête vectorielle.
            target_res = target_chain.run(target_chunks)
            chunk_id = target_chunks[0].metadata.get("chunk_id", "unknown") if target_chunks else "not_found"
            results.append({"pmc_id": pmc_id, "chunk_id": chunk_id, "target": target_res}) ##je passes les chunks à StuffDocumentsChain, elle les concatène en un seul texte → remplace dans le prompt ({context}), envoie ce prompt complet au LLM et récupère la réponse du LLM à la question 
        except Exception as e:
            logger.error(f"Erreur target {pmc_id} : {e}")
            results.append({"pmc_id": pmc_id, "chunk_id": "error", "target": "error"}) #ajouter un élément à la fin d'une liste

        try:
            desc_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(response), output_key="result"),
                document_variable_name="context"
            )
            desc_chunks = vectorstore.similarity_search(response, k=3, filter={"pmc_id": pmc_id})
            desc_res = desc_chain.run(desc_chunks)
            descriptions.append({"pmc_id": pmc_id, "description": desc_res})
        except Exception as e:
            logger.error(f"Erreur description {pmc_id} : {e}")
            descriptions.append({"pmc_id": pmc_id, "description": "error"})

        try:
            domain_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(domain_prompt), output_key="result"),
                document_variable_name="context"
            )
            domain_chunks = vectorstore.similarity_search("biomedical domain", k=3, filter={"pmc_id": pmc_id})
            domain_res = domain_chain.run(domain_chunks)
            chunk_id = domain_chunks[0].metadata.get("chunk_id", "unknown") if domain_chunks else "not_found"
            domains.append({"pmc_id": pmc_id, "chunk_id": chunk_id, "domain": domain_res.strip()})
        except Exception as e:
            logger.error(f"Erreur domain {pmc_id} : {e}")
            domains.append({"pmc_id": pmc_id, "chunk_id": "error", "domain": "error"})

        try:
            effect_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(effect_q), output_key="result"),
                document_variable_name="context"
            )
            effect_chunks = vectorstore.similarity_search(effect_q, k=3, filter={"pmc_id": pmc_id})
            effect_res = effect_chain.run(effect_chunks)
            chunk_id = effect_chunks[0].metadata.get("chunk_id", "unknown") if effect_chunks else "not_found"
            effects.append({"pmc_id": pmc_id, "chunk_id": chunk_id, "effect": effect_res.strip()})
        except Exception as e:
            logger.error(f"Erreur effect {pmc_id} : {e}")
            effects.append({"pmc_id": pmc_id, "chunk_id": "error", "effect": "error"})

        try:
            mech_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(mech_q), output_key="result"),
                document_variable_name="context"
            )
            mech_chunks = vectorstore.similarity_search(mech_q, k=3, filter={"pmc_id": pmc_id})
            mech_res = mech_chain.run(mech_chunks)
            chunk_id = mech_chunks[0].metadata.get("chunk_id", "unknown") if mech_chunks else "not_found"
            mechanisms.append({"pmc_id": pmc_id, "chunk_id": chunk_id, "mechanism": mech_res.strip()})
        except Exception as e:
            logger.error(f"Erreur mechanism {pmc_id} : {e}")
            mechanisms.append({"pmc_id": pmc_id, "chunk_id": "error", "mechanism": "error"})

    # GLOBAL DESCRIPTION
    valid_desc = [d["description"] for d in descriptions if "error" not in d["description"].lower()]
    global_summary = ""
    if valid_desc:
        try:
            full_doc = Document(page_content="\n\n".join(valid_desc))
            global_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(global_prompt), output_key="result"),
                document_variable_name="context"
            )
            global_summary = global_chain.run([full_doc])
        except Exception as e:
            logger.error(f"Erreur global_summary : {e}")
            global_summary = "error"

    return {
        "targets": results,
        "descriptions": descriptions,
        "global_description": global_summary,
        "domains": domains,
        "effects": effects,
        "mechanisms": mechanisms
    }

#ANSI codes pour la couleur
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"

def pretty_print_ansi(title, data): #fonction pour afficher les résultats joliment, avec un titre et les données formatées.
    print(f"\n{title}\n")  # titre avec une ligne vide après
    for item in data: #parcourt chaque dictionnaire dans la liste data, boucle d'affichage (for pmc dans la fonction c la bouxle du traitement)
        pmc = f"{RED_BOLD}pmc_id: {item['pmc_id']}{RESET}"
        other_keys = {k: v for k, v in item.items() if k != "pmc_id"} #extrait les autres éléments du dictionnaire sauf le pmc_id pour afficher le pmc_id à part (en rouge), et tous les autres champs
        print(f"{pmc}, {other_keys}\n")  # saut de ligne après chaque chunk
