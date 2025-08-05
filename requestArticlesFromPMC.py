from loguru import logger
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_pmc_fulltexts_v4(keyword, filtre="", max_results=100):
    full_query = f"{keyword} AND {filtre}" if filtre else keyword
    print(f"Recherche de '{full_query}' sur PubMed Central...")

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pmc",
        "term": full_query,
        "retmax": max_results,
        "retmode": "json"
    }
    r = requests.get(search_url, params=search_params)
    r.raise_for_status()
    idlist = r.json().get("esearchresult", {}).get("idlist", [])

    if not idlist:
        print("Aucun article trouvé.")
        return pd.DataFrame(columns=["keyword", "pmc_id", "fulltext"])

    print(f"{len(idlist)} articles trouvés. Extraction du contenu en cours...")

    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    articles = []

    with tqdm(total=len(idlist), desc="Téléchargement PMC", unit="article", leave=True) as pbar:
        for pmc_id in idlist:
            efetch_params = {
                "db": "pmc",
                "id": pmc_id,
                "retmode": "xml"
            }
            try:
                res = requests.get(efetch_url, params=efetch_params)
                res.raise_for_status()
                soup = BeautifulSoup(res.content, "xml")

                abstract = soup.find("abstract")
                body = soup.find("body")
                if abstract is None and body is None:
                    pbar.update(1)
                    continue

                fulltext = ""
                if abstract:
                    fulltext += f"Abstract: {abstract.get_text(separator=' ', strip=True)}\n\n"
                if body:
                    fulltext += f"Body: {body.get_text(separator=' ', strip=True)}"

                keyword_lower = keyword.lower()
                abstract_text = abstract.get_text(separator=' ', strip=True).lower() if abstract else ""

                intro_text = ""
                discussion_text = ""
                results_text = ""

                for sec in soup.find_all("sec"):
                    title = sec.find("title")
                    if title:
                        title_text = title.get_text(strip=True).lower()
                        sec_text = sec.get_text(separator=' ', strip=True).lower()

                        if "introduction" in title_text:
                            intro_text += sec_text
                        elif "discussion" in title_text:
                            discussion_text += sec_text
                        elif "results" in title_text:
                            results_text += sec_text

                if keyword_lower in abstract_text and (
                    keyword_lower in intro_text or keyword_lower in discussion_text or keyword_lower in results_text
                ):
                    if fulltext.strip():
                        articles.append({
                            "keyword": keyword,
                            "pmc_id": f"PMC{pmc_id}",
                            "fulltext": fulltext
                        })

            except Exception:
                pass  # On ignore les erreurs ici pour garder la barre propre

            pbar.update(1)

    logger.success(f"{len(articles)} articles extraits avec succès.")
    return pd.DataFrame(articles)










df = pd.read_excel('/home/zineb/MOA_mAbs_2025.xlsx')

# Liste / série des mots-clés (INN name)
inn_names = df["INN name"].dropna().unique()

filtre = ""
all_results = []
not_found_article = []

for keyword in inn_names:
    logger.info(f"Traitement du mot-clé : {keyword}")
    try:
        df_pmc = get_pmc_fulltexts_v4(keyword, filtre=filtre, max_results=500)
        if len(df_pmc)==0:
            logger.warning(f"{keyword} not found")
            not_found_article.append(keyword)
        else:
            all_results.append(df_pmc)
                # Concaténer tous les DataFrames extraits en un seul grand DataFrame
            df_all = pd.concat(all_results, ignore_index=True)
            df_all.to_parquet("Full_Articles4All_mAbs.parquet", index=None)
            df_all.to_csv("Full_Articles4All_mAbs.tsv", sep="||", index=None)
        logger.success(f"Traitement du mot-clé : {keyword} OK")
    except Exception as e:
        logger.error(f"Error on {keyword} ==> {e}")
    


logger.error(not_found_article)