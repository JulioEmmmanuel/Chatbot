from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel

from chatbot import chat_bow

app = FastAPI()

@app.get("/rt")
def root():
    return [{"message": "Hello World"}]

class Message(BaseModel):
    msg: str


@app.post("/response")
def get_bot_response(message: Message):
    return [{"res":chat_bow(message)}]

if __name__ == "__main__":
    uvicorn.run("main:app")