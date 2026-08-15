# ============================================================
# RAG PIPELINE
# ============================================================


# Import pickle so we can load our saved FAQ data.
import pickle


# Import FAISS for similarity search.
import faiss


# Import SentenceTransformer to convert text into embeddings.
from sentence_transformers import SentenceTransformer


# Import Ollama so we can communicate with our local Llama model.
import ollama


# ============================================================
# FILE LOCATIONS
# ============================================================

# This is the location of our FAISS vector database.
INDEX_FILE = "vector_store/faq.index"


# This is the location of our saved FAQ information.
FAQ_FILE = "vector_store/faq_data.pkl"


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

# Load the same embedding model that we used
# when creating the FAISS database.
#
# all-MiniLM-L6-v2 converts text into numerical vectors.
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

# Read the saved FAISS vector database from disk.
index = faiss.read_index(
    INDEX_FILE
)


# ============================================================
# LOAD FAQ DATA
# ============================================================

# Open our saved FAQ data file.
#
# "rb" means read the file as binary data.
with open(
    FAQ_FILE,
    "rb"
) as file:

    # Convert the saved binary data back into Python objects.
    faq_data = pickle.load(file)


# ============================================================
# RETRIEVE RELEVANT FAQs
# ============================================================

def retrieve_faqs(
    user_question,
    top_k=3
):
    """
    Find the most relevant FAQs for the user's question.

    user_question = question asked by the user.

    top_k = number of FAQs we want to retrieve.
    """

    # --------------------------------------------------------
    # STEP 1: Convert user's question into a vector
    # --------------------------------------------------------

    # Sentence Transformer converts the user's question
    # into a numerical embedding.
    #
    # normalize_embeddings=True makes cosine similarity
    # search work correctly with our FAISS setup.
    question_embedding = model.encode(
        [user_question],
        normalize_embeddings=True
    )


    # --------------------------------------------------------
    # STEP 2: Convert vector to float32
    # --------------------------------------------------------

    # FAISS expects vectors in float32 format.
    question_embedding = question_embedding.astype(
        "float32"
    )


    # --------------------------------------------------------
    # STEP 3: Search FAISS
    # --------------------------------------------------------

    # Search our FAISS database.
    #
    # question_embedding = user's vector.
    # top_k = how many similar FAQs we want.
    #
    # scores = similarity scores.
    # indexes = positions of matching FAQs.
    scores, indexes = index.search(
        question_embedding,
        top_k
    )


    # --------------------------------------------------------
    # STEP 4: Store retrieved FAQs
    # --------------------------------------------------------

    # Create an empty list to store our results.
    results = []


    # Go through each FAQ returned by FAISS.
    for position, faq_index in enumerate(
        indexes[0]
    ):

        # FAISS can return -1 when there is no result.
        # We ignore such results.
        if faq_index < 0:
            continue


        # Get the FAQ corresponding to this index.
        faq = faq_data[faq_index]


        # Create a clean result dictionary.
        result = {

            # Store FAQ category.
            "category": faq["category"],

            # Store original FAQ question.
            "question": faq["question"],

            # Store FAQ answer.
            "answer": faq["answer"],

            # Store similarity score.
            "score": float(
                scores[0][position]
            )
        }


        # Add this FAQ to our results.
        results.append(
            result
        )


    # Return the retrieved FAQs.
    return results


# ============================================================
# GENERATE ANSWER USING LOCAL LLM
# ============================================================

def generate_answer(
    user_question
):
    """
    Complete RAG process:

    1. Retrieve relevant FAQs using Sentence Transformers + FAISS.
    2. Send those FAQs to Llama through Ollama.
    3. Return Llama's answer.
    """


    # --------------------------------------------------------
    # STEP 1: RETRIEVE TOP 3 FAQs
    # --------------------------------------------------------

    # Ask FAISS to find the 3 most relevant FAQs.
    retrieved_faqs = retrieve_faqs(
        user_question,
        top_k=3
    )


    # --------------------------------------------------------
    # STEP 2: CREATE CONTEXT
    # --------------------------------------------------------

    # Start with an empty context.
    context = ""


    # Add every retrieved FAQ to the context.
    for number, faq in enumerate(
        retrieved_faqs,
        start=1
    ):

        # Add FAQ information to our context.
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

    # Tell Llama exactly how it should answer.
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
    # STEP 4: SEND CONTEXT TO LOCAL LLAMA
    # --------------------------------------------------------

    # Send our prompt to Llama 3.2 3B.
    #
    # "ollama.chat()" communicates with the Ollama application
    # running on our computer.
    response = ollama.chat(

        # Tell Ollama which local model to use.
        model="llama3.2:3b",

        # Give Llama the instructions and context.
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # Lower temperature makes the answer
        # more predictable and less creative.
        options={
            "temperature": 0.2
        }
    )


    # --------------------------------------------------------
    # STEP 5: EXTRACT LLAMA'S ANSWER
    # --------------------------------------------------------

    # Ollama returns the generated text inside:
    #
    # response["message"]["content"]
    #
    # So we extract that text here.
    answer = response["message"]["content"]


    # --------------------------------------------------------
    # STEP 6: RETURN RESULT
    # --------------------------------------------------------

    # Return both:
    #
    # answer = Llama's final response.
    #
    # sources = FAQs that FAISS retrieved.
    return {
        "answer": answer,
        "sources": retrieved_faqs
    }