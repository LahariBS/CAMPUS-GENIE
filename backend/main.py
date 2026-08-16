
# Import FastAPI to create our backend API.
from fastapi import FastAPI

# Import BaseModel to define the format of incoming data.
from pydantic import BaseModel

# Import CORS middleware so our HTML/JavaScript frontend
# can communicate with the FastAPI backend.
from fastapi.middleware.cors import CORSMiddleware

# Import our RAG function.
from backend.rag_pipeline import generate_answer


# Create the FastAPI application.
app = FastAPI(
    title="College Internship RAG Chatbot"
)


# Allow the frontend to communicate with our backend.
app.add_middleware(
    CORSMiddleware,

    # Allow requests from our frontend.
     allow_origins=["*"],
    

    # Allow cookies if needed.
    allow_credentials=False,

    # Allow GET, POST and other HTTP methods.
    allow_methods=["*"],

    # Allow all request headers.
    allow_headers=["*"],
)


# Define the format of the user's question.
class QuestionRequest(BaseModel):

    # The frontend will send the question using this field.
    question: str


# Create a simple test endpoint.
@app.get("/")
def home():

    # Return a message when we visit the backend.
    return {
        "message": "College Internship RAG Chatbot API is running!"
    }


# Create the chatbot endpoint.
@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # Get the question sent by the frontend.
    question = request.question

    # Send the question to our RAG pipeline.
    result = generate_answer(
        question
    )

    # Send the RAG result back to the frontend.
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }