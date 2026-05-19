# LLM_NETFLIX - Chatbot RAG local sur le catalogue Netflix

## Présentation

Ce projet est un chatbot capable de répondre à des questions sur un catalogue Netflix local à partir du fichier `data/netflix_titles.csv`.

Il utilise une architecture RAG, pour *Retrieval-Augmented Generation* :

1. Le dataset Netflix est transformé en documents.
2. Les documents sont convertis en embeddings avec Ollama.
3. Les embeddings sont stockés dans ChromaDB.
4. Quand l'utilisateur pose une question, le projet récupère les documents les plus pertinents.
5. Le modèle local `llama3.2:1b` répond à partir du contexte récupéré.

Le projet fonctionne entièrement en local et n'utilise pas d'API externe.

## Stack utilisée

- Python
- uv pour l'environnement d'exécution
- LangChain pour la chaîne RAG
- Ollama pour les modèles locaux
- `llama3.2:1b` comme modèle de langage
- `nomic-embed-text` comme modèle d'embeddings
- ChromaDB comme base vectorielle persistante
- pandas pour lire le CSV
- Streamlit pour l'interface web

## Structure du projet

```text
.
├── data/
│   └── netflix_titles.csv
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   ├── chain.py
│   ├── app.py
│   └── web_app.py
├── requirements.txt
├── Fonctionnement.txt
├── README.md
└── .gitignore
```

Le dossier `chroma_db/` est généré localement après l'ingestion et n'est pas versionné.

## Installation

Installer les dépendances :

```powershell
uv pip install -r requirements.txt
```

Télécharger les modèles Ollama :

```powershell
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

## Ingestion des données

Avant de lancer le chatbot, il faut créer la base vectorielle ChromaDB :

```powershell
uv run python src/ingest.py
```

Le script :

- lit `data/netflix_titles.csv`
- prépare un document par titre
- génère les embeddings
- stocke les données dans `chroma_db/`
- évite la double ingestion si la collection existe déjà

La collection ChromaDB utilisée s'appelle :

```text
netflix_catalog
```

## Lancer le chatbot CLI

```powershell
uv run python src/app.py
```

Pour quitter :

```text
/quit
```

## Lancer l'interface web

```powershell
uv run streamlit run src/web_app.py
```

Puis ouvrir :

```text
http://localhost:8501
```

Si Streamlit garde un ancien cache ou si le port pose problème :

```powershell
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
uv run streamlit run src/web_app.py --server.port 8502
```

Puis ouvrir :

```text
http://localhost:8502
```

## Exemples de questions

Le frontend est en anglais, car le modèle `llama3.2:1b` répond mieux en anglais.

Exemples :

```text
Give me the description of League of Legends Origins
What do you know about Fate/Zero?
Who directed Bird Box?
all pokemon movies or series that are available
all titles that start with the legend of
give me all the nova movies available
movies with director containing John
movies with cast containing Sandra
movies released in 2020
series from Japan released in 2020
list movies containing love
```

## Fonctionnalités principales

- Réponses sur les films et séries du catalogue Netflix local
- Recherche vectorielle avec ChromaDB
- Questions de résumé ou description avec le modèle local
- Filtres structurés pour les listes
- Recherche par titre, préfixe de titre, réalisateur, acteurs, pays, genre, année, type et description
- Interface CLI
- Interface web Streamlit

## Limites connues

- Le modèle `llama3.2:1b` est léger et peut parfois mal suivre un long contexte.
- Les questions de liste sont donc filtrées côté Python pour éviter que le modèle oublie des titres.
- Le modèle n'est pas entraîné sur le dataset : il lit seulement les documents récupérés au moment de la question.
- La base ChromaDB doit être générée localement avec `src/ingest.py`.

## Commandes utiles

```powershell
uv pip install -r requirements.txt
uv run python src/ingest.py
uv run python src/app.py
uv run streamlit run src/web_app.py
```

