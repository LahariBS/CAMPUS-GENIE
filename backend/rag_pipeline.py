# ============================================================
# RAG PIPELINE
# ============================================================

import pickle
import os

import faiss
import numpy as np

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)


# ============================================================
# FILE LOCATIONS
# ============================================================

INDEX_FILE = "vector_store/faq.index"

FAQ_FILE = "vector_store/faq_data.pkl"

VECTORIZER_FILE = "vector_store/tfidf_vectorizer.pkl"


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

with open(
    VECTORIZER_FILE,
    "rb"
) as file:

    vectorizer = pickle.load(file)


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

index = faiss.read_index(
    INDEX_FILE
)


# ============================================================
# LOAD FAQ DATA
# ============================================================

with open(
    FAQ_FILE,
    "rb"
) as file:

    faq_data = pickle.load(file)


# ============================================================
# RETRIEVE RELEVANT FAQs
# ============================================================

def retrieve_faqs(
    user_question,
    top_k=3
):

    # --------------------------------------------------------
    # STEP 1: Convert question into TF-IDF vector
    # --------------------------------------------------------

    question_embedding = vectorizer.transform(
        [user_question]
    )


    # --------------------------------------------------------
    # STEP 2: Convert to float32
    # --------------------------------------------------------

    question_embedding = question_embedding.toarray().astype(
        "float32"
    )


    # --------------------------------------------------------
    # STEP 3: Normalize vector
    # --------------------------------------------------------

    norm = np.linalg.norm(
        question_embedding,
        axis=1,
        keepdims=True
    )

    norm[norm == 0] = 1

    question_embedding = question_embedding / norm


    # --------------------------------------------------------
    # STEP 4: Search FAISS
    # --------------------------------------------------------

    scores, indexes = index.search(
        question_embedding,
        top_k
    )


    # --------------------------------------------------------
    # STEP 5: Store retrieved FAQs
    # --------------------------------------------------------

    results = []


    for position, faq_index in enumerate(
        indexes[0]
    ):

        if faq_index < 0:
            continue


        faq = faq_data[faq_index]


        result = {

            "category": faq["category"],

            "question": faq["question"],

            "answer": faq["answer"],

            "score": float(
                scores[0][position]
            )
        }


        results.append(
            result
        )


    return results


# ============================================================
# GENERATE ANSWER USING HUGGING FACE
# ============================================================

def generate_answer(
    user_question
):

    # --------------------------------------------------------
    # STEP 1: RETRIEVE TOP 3 FAQs
    # --------------------------------------------------------

    retrieved_faqs = retrieve_faqs(
        user_question,
        top_k=3
    )


    # --------------------------------------------------------
    # STEP 2: CREATE CONTEXT
    # --------------------------------------------------------

    context = ""


    for number, faq in enumerate(
        retrieved_faqs,
        start=1
    ):

        context += (
            f"\nFAQ {number}\n"
            f"Category: {faq['category']}\n"
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}\n"
            f"Similarity: {faq['score']:.3f}\n"
        )


    # --------------------------------------------------------
    # STEP 3: CREATE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a helpful College Admission and Internship FAQ assistant.

You MUST answer the user's question using ONLY the FAQ context
provided below.

Do NOT invent information.

If the answer cannot be found in the FAQ context, say:

"Sorry, this information is not available in the current FAQ knowledge base."

Keep the answer clear, simple and concise.

================ FAQ CONTEXT ================
{context}

================ USER QUESTION ================
{user_question}

================ ANSWER ================
"""


    # --------------------------------------------------------
    # STEP 4: SEND TO HUGGING FACE
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b:fastest",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=500
    )


    # --------------------------------------------------------
    # STEP 5: EXTRACT ANSWER
    # --------------------------------------------------------

    answer = response.choices[0].message.content


    # --------------------------------------------------------
    # STEP 6: RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": retrieved_faqs
    }