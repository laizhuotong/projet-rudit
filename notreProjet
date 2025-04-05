# ⚠️ Ignorer les avertissements
import warnings

# Ignorer les FutureWarning de huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import os
import re
import math
import hashlib
import uuid
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# ⚠️ Charger le modèle multilingue
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

from lxml import etree
import numpy as np

DB_PATH = "./chroma_db"

class EruditAISystem:
    def __init__(self):
        # Paramètres de configuration
        self.NSMAP = {
            'erudit': 'http://www.erudit.org/xsd/article',
            'xlink': 'http://www.w3.org/1999/xlink'
        }
        self.MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
        self.XML_FOLDER = "./xml_articles"
        self.MAX_TEXT_LENGTH = 8000
        self.EMBEDDING_DIM = 384  # Dimension fixe

        # Initialiser les composants
        self.model = SentenceTransformer(self.MODEL_NAME)

        # self.chroma_client = chromadb.Client()
        self.DB_PATH = DB_PATH
        if os.path.exists(self.DB_PATH):
            print(f"Chargement de la base de données existante depuis {self.DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(path=self.DB_PATH)
        else:
            print("Création d'une nouvelle base de données ChromaDB")
            self.chroma_client = chromadb.PersistentClient(path=self.DB_PATH)

        # Charger ou créer la base de données persistante
        print(f"Chargement de la base de données depuis {self.DB_PATH}")
        self.chroma_client = chromadb.PersistentClient(path=self.DB_PATH)

        # Vérifier si la collection existe déjà
        existing_collections = self.chroma_client.list_collections()
        collection_name = "erudit_articles"

        self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        if collection_name in existing_collections:
            print(f"Collection existante détectée: {collection_name}")
        else:
            print(f"Création d'une nouvelle collection: {collection_name}")
            self.process_xml()

        

    def _generate_id(self, folder_path, metadata):
        doi = metadata.get("doi")
        if doi:
            return doi  # Utilise le DOI directement comme ID s'il existe
        return f"{folder_path}_{uuid.uuid4().hex[:8]}"
    
    def _extract_metadata(self, root):
        """Extraire les métadonnées du XML"""
        year_elem = root.find(".//erudit:pub/erudit:annee", namespaces=self.NSMAP)
        return {
            "title": root.findtext(".//erudit:titre", namespaces=self.NSMAP, default="Sans titre").strip(),
            "author": self._get_authors(root),
            "year": year_elem.text[:4] if year_elem is not None else "",
            "language": root.get("lang", "").lower(),
            "doi": root.findtext(".//erudit:idpublic[@scheme='doi']", namespaces=self.NSMAP, default=""),
            "keywords": ";".join([kw.text for kw in root.findall(".//erudit:motcle", self.NSMAP) if kw.text])
        }

    def _get_authors(self, root):
        """Extraire les informations des auteurs"""
        authors = []
        for author_elem in root.findall(".//erudit:auteur", self.NSMAP):
            firstname = author_elem.findtext(".//erudit:prenom", "", self.NSMAP) or ""
            lastname = author_elem.findtext(".//erudit:nomfamille", "", self.NSMAP) or ""
            authors.append(f"{firstname} {lastname}".strip())
        return "、".join(authors) if authors else "Auteur inconnu"

    # def _extract_content(self, root):
    #     """Extraire le contenu structuré"""
    #     # Extraire le résumé
    #     abstract = ""
    #     for resume in root.findall(".//erudit:resume[@typeresume='resume']", self.NSMAP):
    #         if resume.get("lang") == "fr":
    #             abstract = " ".join([p.text for p in resume.findall(".//erudit:alinea", self.NSMAP) if p.text])
    #             break

    #     # Extraire les sections du texte principal
    #     sections = []
    #     for section in root.findall(".//erudit:section1", self.NSMAP):
    #         title = section.findtext(".//erudit:titre", "", self.NSMAP) or ""
    #         paras = [p.text for p in section.findall(".//erudit:alinea", self.NSMAP) if p.text]
    #         if title and paras:
    #             sections.append(f"【{title}】\n{' '.join(paras)}")

    #         full_text = f"{abstract}\n\n" + "\n\n".join(sections)
    #         return re.sub(r'\s+', ' ', full_text)[:self.MAX_TEXT_LENGTH]
    def _extract_content(self, root):
        """Extraire le contenu structuré"""
        try:
            abstract = ""
            # Résumé de l'analyse
            for resume in root.findall(".//erudit:resume[@typeresume='resume']", self.NSMAP):
                if resume.get("lang") == "fr":
                    abstract = " ".join([p.text for p in resume.findall(".//erudit:alinea", self.NSMAP) if p.text])
                    break

            # S'adapter à deux structures de texte :
            # 1️⃣ Ancienne structure <corps><texte><alinea>
            text_elements_1 = root.findall(".//erudit:corps/erudit:texte/erudit:alinea", self.NSMAP)
            
            # 2️⃣ Nouvelle stucture <corps><section1><para><alinea>
            text_elements_2 = root.findall(".//corps//section1//para//alinea", self.NSMAP)

            paragraphs = [p.text.strip() for p in (text_elements_1 + text_elements_2) if p.text]  
            
            full_text = "\n\n".join(paragraphs)

            # Assurez-vous que le texte renvoyé n'est pas Aucun
            return re.sub(r'\s+', ' ', full_text)[:self.MAX_TEXT_LENGTH] if full_text.strip() else "Texte indisponible"

        except Exception as e:
            print(f"Erreur lors de l'extraction du contenu: {str(e)}")
            return "Texte indisponible"


    def _validate_embedding(self, embedding):
        """Valider l'efficacité du vecteur d'embedding"""
        if len(embedding) != self.EMBEDDING_DIM:
            raise ValueError(f"Erreur de dimension: attendu {self.EMBEDDING_DIM}, obtenu {len(embedding)}")
        if any(math.isnan(x) for x in embedding):
            raise ValueError("Contient des valeurs NaN")
        return True

    def process_xml(self):
        """Traiter les fichiers XML et les stocker dans la base de données"""
        xml_files = list(Path(self.XML_FOLDER).glob("**/*.xml"))
        print(f"{len(xml_files)} fichiers XML trouvés")

        pattern = re.compile(r"^ERUDITXSD\d{3}\.xml$")  
        existing_ids = set(self.collection.get()["ids"])  # Enregistrer les IDs existants
        success_count = 0
        for xml_file in tqdm(xml_files, desc="Traitement des fichiers XML"):
            try:
                if not pattern.match(xml_file.name):
                    print(f"Ignoré (nom incorrect): {xml_file}")
                    continue

                # Parser le XML
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(str(xml_file), parser)
                root = tree.getroot()

                # Extraire les données
                metadata = self._extract_metadata(root)
                full_text = self._extract_content(root)
                
                if not full_text.strip():
                    print(f"Fichier vide ignoré: {xml_file.name}")
                    continue

                # Générer l'ID
                folder_path = "/".join(Path(xml_file).parts[-3:])
                doc_id = metadata.get("doi") or self._generate_id(folder_path, metadata)

                if doc_id in existing_ids:
                    print(f"Document avec DOI {doc_id} existant, mise à jour ignoré: {xml_file.name}")
                    continue

                # Générer l'embedding
                embedding = self.model.encode(full_text, normalize_embeddings=True)
                embedding_list = embedding.tolist()
                print("Embedding (first 5 values):", embedding_list[:5])
                print(f"Text preview: {full_text[:100]}")
                self._validate_embedding(embedding_list)

                # Stocker dans la base de données
                self.collection.add(
                    embeddings=[embedding_list],
                    documents=[full_text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                success_count += 1

            except Exception as e:
                print(f"\nÉchec du traitement: {xml_file} with error: {str(e)}")
                print("Skipping to the next file...")
                continue

        # Validation finale
        print("\n=== Validation de l'intégrité des données ===")
        print(f"Fichiers traités: {len(xml_files)}")
        print(f"Documents stockés avec succès: {success_count}")
        print(f"Documents échoués: {len(xml_files) - success_count}")
        print(f"Documents total: {self.collection.count()}")
        print("=== Fin de la validation ===")

    def semantic_search(self, query, lang_filter=None, top_k=5):
        """Fonction de recherche sémantique"""
        try:
            if not query.strip():
                raise ValueError("Le contenu de la recherche ne peut pas être vide")

            total_docs = self.collection.count()
            if total_docs == 0:
                return []

            top_k = min(top_k, total_docs)
            
            # Générer l'embedding de la requête avec le modèle multilingue
            query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
            
            # Exécuter la requête, sans restriction de langue
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                # ⚠️ where={"language": lang_filter} if lang_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_results(results)
        except Exception as e:
            print(f"\nÉchec de la recherche: {str(e)}")
            return []

    def _format_results(self, results):
        """Formater les résultats"""
        formatted = []
        
        if isinstance(results, dict):
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
        else:
            documents = []
            metadatas = []
            distances = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append({
                "title": meta.get("title", "Sans titre"),
                "author": meta.get("author", "Auteur inconnu"),
                "year": meta.get("year", ""),
                # ⚠️ Ajouté pour aider l'utilisateur à comprendre pourquoi certains documents sont retournés.
                "language": meta.get("language", "Langue inconnue"),  # Afficher la langue du document
                "similarity": f"{1 - dist:.2f}" if dist is not None else "N/A",
                "excerpt": (doc[:200] + "...") if doc and len(doc) > 200 else doc or "",
                "doi": meta.get("doi", "")
            })
        return formatted

    def get_recommendations(self, doc_id, top_k=5):
        # Vérifier l'existence du document
        all_ids = self.collection.get()["ids"]
        if doc_id not in all_ids:
            raise ValueError(f"Le document {doc_id} n'existe pas")

        # Obtenir les données du document
        existing_data = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "documents", "metadatas"]
        )

        if existing_data["embeddings"].size == 0:
            raise ValueError("Aucun vecteur d'embedding valide")
        self._validate_embedding(existing_data["embeddings"][0])

        # Exécuter la requête
        results = self.collection.query(
            query_embeddings=existing_data["embeddings"],
            n_results=top_k + 1,  # +1 pour exclure l'article lui-même
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        valid_results = []

        for i in range(len(documents)):
            result_doc_id = metadatas[i].get("doi") or ""  # 如果你用其他 ID 生成方式，也可以改成 metadatas[i].get("id", "")
            if result_doc_id != doc_id:
                valid_results.append((documents[i], metadatas[i], distances[i]))

        return self._format_results({
            "documents": [[x[0] for x in valid_results]],
            "metadatas": [[x[1] for x in valid_results]],
            "distances": [[x[2] for x in valid_results]]
        })


    def show_document_list(self, max_items=10):
        """Afficher la liste des documents avec des numéros"""
        docs = self.collection.get()
        if not docs["ids"]:
            print("La base de données est vide")
            return []

        print("\nListe actuelle des documents:")

        # ⬇️ Trier par ID, garantissant un ordre cohérent (quel que soit le système d'exploitation)
        sorted_items = sorted(zip(docs["ids"], docs["metadatas"]), key=lambda x: x[0])
        
        display_list = []
        for idx, (doc_id, meta) in enumerate(sorted_items, 1):
            if idx > max_items:
                print(f"... (total {len(docs['ids'])} documents)")
                break
            title = meta.get("title", "Sans titre")[:100]
            print(f"[{idx}] {title}... (ID: {doc_id})")
            display_list.append(doc_id)

        return display_list


if __name__ == "__main__":
    erudit_ai = EruditAISystem()
    # erudit_ai.process_xml()

    # Interface interactive
    while True:
        print("="*50)
        print("Système de recherche de documents académiques")
        print("1. Recherche sémantique")
        print("2. Recommandation intelligente")
        print("3. Quitter")
        choice = input("Choisissez une opération (1-3): ").strip()

        if choice == "1":
            print("\n=== Recherche sémantique ===")
            query = input("Entrez le contenu de la recherche: ").strip()
            lang = input("Filtrer par langue (optionnel, par exemple fr/en): ").strip().lower() or None
            results = erudit_ai.semantic_search(query, lang_filter=lang)
            
            if results:
                print(f"\n{len(results)} résultats trouvés:")
                for idx, res in enumerate(results, 1):
                    print(f"\n[{idx}] {res['title']}")
                    print(f"   Auteur: {res['author']}")
                    print(f"   Année: {res['year']} | Similarité: {res['similarity']}")
                    print(f"   DOI: {res['doi']}")
                    print(f"   Résumé: {res['excerpt']}")
            else:
                print("\nAucun résultat trouvé")

        elif choice == "2":
            print("\n=== Recommandation intelligente ===")
            display_ids = erudit_ai.show_document_list()
            
            if not display_ids:
                continue
                
            # try:
            choice = input("Entrez le numéro du document (ou 'q' pour quitter): ").strip()
            if choice.lower() == 'q':
                continue
            # choice_idx = int(choice) - 1
            # if choice_idx < 0 or choice_idx >= len(display_ids):
            #     raise ValueError
            # doc_id = display_ids[choice_idx]
            if choice.isdigit():  # S'il s'agit d'un nombre pur, interrogez par numéro de séquence
                choice_idx = int(choice) - 1
                if choice_idx < 0 or choice_idx >= len(display_ids):
                    print("\nNuméro invalide, veuillez réessayer.")
                    continue
                doc_id = display_ids[choice_idx]
            else:  # S'il s'agit d'un format DOI, interrogez directement
                doc_id = choice.strip()
                #     if doc_id not in display_ids:
                #     print("\nID de document invalide, veuillez réessayer.")
                # continue
            
            recommendations = erudit_ai.get_recommendations(doc_id)
            if recommendations:
                print("\nRésultats de la recommandation:")
                for idx, rec in enumerate(recommendations, 1):
                    print(f"{idx}. {rec['title']}")
                    print(f"   Similarité: {rec['similarity']} | Auteur: {rec['author']}")
            else:
                print("\nAucune recommandation trouvée")
            # except:
            #     print("\nEntrée invalide, veuillez utiliser un numéro de la liste")

        elif choice == "3":
            print("\nQuitter le système")
            break

        else:
            print("\nEntrée invalide, veuillez réessayer")

        input("\nAppuyez sur Entrée pour continuer...")
        os.system('cls' if os.name == 'nt' else 'clear')
