from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import GROQ_API_KEY

SYSTEM_PROMPT = """Tu rédiges un message WhatsApp pour un club de randonnée (Rando Club) \
qui informe un membre que sa participation à une sortie est acceptée.

Règles de format strictes :
- Formatage WhatsApp uniquement : *gras* pour les mots-clés, pas de markdown (pas de **, pas de _italique_)
- Utilise des emojis pertinents (👋 📅 📍 📏 ⏱ ⛰ 🗺 👥 ✅ 🟢) en début de ligne, comme dans l'exemple
- Ton chaleureux et concis, une info par ligne, pas de paragraphe
- N'invente aucune information : n'utilise que les données fournies, saute les lignes optionnelles si la donnée est vide
- Termine toujours par une consigne pour confirmer sa présence (répondre OUI) puis le lien du groupe WhatsApp
- Réponds uniquement avec le message final, sans commentaire ni guillemets autour
"""

USER_PROMPT = """Données de la sortie :
- Titre : {title}
- Date : {date}
- Heure : {time}
- Lieu de départ : {location}
- Distance : {distance}
- Durée : {duration}
- Dénivelé : {elevation}
- Lien itinéraire : {maps_url}
- Places restantes : {places_left}
- Lien du groupe WhatsApp : {group_link}
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)

_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.4)

whatsapp_message_chain = _prompt | _llm | StrOutputParser()

