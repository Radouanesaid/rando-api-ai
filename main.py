from fastapi import FastAPI
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from chains import rag
from chains.whatsapp_message import whatsapp_message_chain

langfuse_handler = CallbackHandler()

app = FastAPI(title="api-ai")


@app.get("/health")
def health():
    return {"status": "ok"}


class WhatsappMessageRequest(BaseModel):
    title: str
    date: str
    time: str
    location: str
    distance: str = ""
    duration: str = ""
    elevation: str = ""
    maps_url: str = ""
    places_left: str = ""
    group_link: str = ""


class WhatsappMessageResponse(BaseModel):
    message: str


@app.post("/generate/whatsapp-message", response_model=WhatsappMessageResponse)
def generate_whatsapp_message(payload: WhatsappMessageRequest):
    message = whatsapp_message_chain.invoke(
        payload.model_dump(), config={"callbacks": [langfuse_handler]}
    )
    return WhatsappMessageResponse(message=message)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    result = rag.ask(payload.question, callbacks=[langfuse_handler])
    return AskResponse(**result)
