# api-ai

Backend IA pour **Rando Club** (club de randonnée/vélo à Paris et en Île-de-France). Expose une API FastAPI qui combine :

- un **chatbot RAG** (Retrieval-Augmented Generation) répondant à des questions sur le club à partir d'un corpus indexé dans **ChromaDB**,
- un **générateur de messages WhatsApp** pour confirmer à un membre sa participation à une sortie.

Les deux fonctionnalités sont implémentées comme des **chaînes LangChain**, tracées via **Langfuse**, et servies par **Groq** (LLM) et **Google Generative AI** (embeddings).

## Architecture

```
                         ┌─────────────────────┐
                         │      FastAPI         │
                         │      (main.py)       │
                         └─────────┬────────────┘
                    ┌──────────────┴──────────────┐
                    │                              │
          POST /ask                      POST /generate/whatsapp-message
                    │                              │
                    ▼                              ▼
          ┌──────────────────┐          ┌───────────────────────┐
          │  chains/rag.py    │          │ chains/whatsapp_       │
          │                    │          │ message.py             │
          │ 1. embed question  │          │ prompt | ChatGroq |    │
          │ 2. query Chroma    │          │ StrOutputParser        │
          │ 3. build context   │          └───────────────────────┘
          │ 4. prompt | LLM    │
          │ 5. parse output    │
          └─────────┬──────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌───────────────────┐
│ Google Gemini     │    │ ChromaDB           │
│ embedding-001      │    │ (chroma_data/)     │
│ (question → vector)│    │ collection:        │
└─────────────────┘    │  "rando_club"       │
                        └───────────────────┘
                                  ▲
                                  │ upsert (embeddings + docs + metadata)
                                  │
                        ┌───────────────────┐
                        │ index_corpus.py     │
                        │ reads corpus/*.json │
                        └───────────────────┘

Every chain invocation is wrapped with a Langfuse CallbackHandler for tracing/observability.
```

### Composants

| Fichier | Rôle |
|---|---|
| [main.py](main.py) | Application FastAPI : définit les endpoints `/health`, `/ask`, `/generate/whatsapp-message`. Instancie le `CallbackHandler` Langfuse partagé et l'injecte dans chaque appel de chaîne. |
| [config.py](config.py) | Charge les variables d'environnement depuis `.env.{APP_ENV}` (`local`, `prod`, `rec`, ...) via `python-dotenv`. Centralise les clés API (Groq, Gemini, Langfuse). |
| [chains/rag.py](chains/rag.py) | Chaîne RAG : retrieval Chroma + prompt système strict + LLM Groq. Expose `ask(question)`. |
| [chains/whatsapp_message.py](chains/whatsapp_message.py) | Chaîne de génération de message WhatsApp formaté (emojis, gras WhatsApp) à partir des détails d'une sortie. |
| [index_corpus.py](index_corpus.py) | Script CLI d'indexation : lit `corpus/*.json`, calcule les embeddings Gemini, et fait un `upsert` dans la collection Chroma persistante. Inclut aussi des requêtes de test manuelles. |
| [corpus/](corpus) | Données sources du RAG, au format JSON (`destinations.json`, `history.json`). Chaque entrée a un `id`, un `type` (`destination` / `history`) et un `text`. |
| [chroma_data/](chroma_data) | Base ChromaDB persistante (SQLite + fichiers HNSW binaires). Générée par `index_corpus.py`, lue par `chains/rag.py`. Ne pas éditer à la main. |

## Flux RAG en détail (`/ask`)

1. **Indexation (offline, `index_corpus.py`)**
   - Charge toutes les entrées de `corpus/*.json`.
   - Calcule un embedding par document avec `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`).
   - `upsert` dans la collection Chroma `rando_club` : `id`, vecteur, texte brut, et métadonnée `type`.
   - Stockage persistant sur disque dans `chroma_data/` (client `chromadb.PersistentClient`).

2. **Requête (runtime, `chains/rag.py:ask`)**
   - La question utilisateur est embeddée avec le même modèle Gemini.
   - `collection.query(...)` renvoie les `TOP_K=3` documents les plus proches (similarité vectorielle).
   - Les documents récupérés sont concaténés en un bloc `context` (`- [type] texte`).
   - Un prompt système strict (répondre uniquement à partir du contexte, en français, ne rien inventer) est envoyé à `ChatGroq` (`llama-3.3-70b-versatile`).
   - La réponse est retournée avec la liste des `sources` (ids des documents utilisés), pour traçabilité.

Ce pattern garantit que le modèle ne répond qu'avec des informations issues du corpus indexé (pas d'hallucination sur les dates/lieux du club), et que chaque réponse est vérifiable via ses sources.

## Flux de génération WhatsApp (`/generate/whatsapp-message`)

- Prend en entrée les détails d'une sortie (titre, date, heure, lieu, distance, dénivelé, lien, places restantes, lien de groupe).
- Une chaîne simple `prompt | ChatGroq | StrOutputParser` génère un message formaté façon WhatsApp (`*gras*`, emojis, une info par ligne), en omettant les champs optionnels vides.
- Aucune recherche/RAG ici — uniquement du templating piloté par prompt.

## Observabilité

Chaque appel de chaîne (`rag.ask`, `whatsapp_message_chain.invoke`) reçoit un `langfuse.langchain.CallbackHandler` commun, instancié une fois dans `main.py`. Cela envoie automatiquement les traces (prompts, latence, tokens, sorties) vers Langfuse (cloud ou self-hosted, configurable via `LANGFUSE_HOST`).

## Stack technique

- **FastAPI** — API HTTP
- **LangChain** (`langchain`, `langchain-groq`, `langchain-google-genai`) — orchestration des chaînes prompt → LLM → parser
- **Groq** (`llama-3.3-70b-versatile`) — inférence LLM
- **Google Generative AI** (`gemini-embedding-001`) — embeddings pour le RAG
- **ChromaDB** — base vectorielle persistante locale
- **Langfuse** — tracing/observabilité des chaînes LLM
- **uv** — gestion des dépendances/environnement Python (`pyproject.toml`, `uv.lock`)

## Installation

```bash
uv sync
cp .env.example .env.local
# renseigner GROQ_API_KEY, GEMINI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

`APP_ENV` détermine le fichier `.env` chargé (`.env.local`, `.env.prod`, `.env.rec`, ...), par défaut `local`.

## Indexer le corpus

```bash
uv run python index_corpus.py
```

À relancer à chaque modification de `corpus/*.json` pour mettre à jour la collection Chroma (`chroma_data/`).

## Lancer l'API

```bash
uv run uvicorn main:app --reload --port 8081
```

## Endpoints

### `GET /health`
Vérification simple de disponibilité.

### `POST /ask`
```json
{ "question": "une rando facile" }
```
Réponse :
```json
{ "answer": "...", "sources": ["etretat-aout-2025", "..."] }
```

### `POST /generate/whatsapp-message`
```json
{
  "title": "Randonnee a Fontainebleau",
  "date": "samedi 25 juillet",
  "time": "09h00",
  "location": "Gare de Fontainebleau-Avon",
  "distance": "15 km",
  "duration": "4h30",
  "elevation": "320 m",
  "maps_url": "https://maps.google.com/?daddr=Fontainebleau",
  "places_left": "4",
  "group_link": "https://chat.whatsapp.com/exemple123"
}
```
Réponse :
```json
{ "message": "👋 *Randonnée à Fontainebleau*\n📅 ..." }
```

Voir [test_ask.sh](test_ask.sh) et [test_payload.json](test_payload.json) pour des exemples d'appels.
