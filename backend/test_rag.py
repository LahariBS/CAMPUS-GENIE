# Import the complete RAG function.
# This function will retrieve FAQs using FAISS
# and then send them to Llama.
from rag_pipeline import generate_answer


# ------------------------------------------------------------
# TEST QUESTION
# ------------------------------------------------------------

# This is the question we want our chatbot to answer.
question = "What documents do I need for college admission?"


# ------------------------------------------------------------
# RUN COMPLETE RAG PIPELINE
# ------------------------------------------------------------

# Send the question through:
#
# Sentence Transformer
#        ↓
# FAISS
#        ↓
# Top 3 FAQs
#        ↓
# Llama 3.2 3B
#        ↓
# Final answer
result = generate_answer(
    question
)


# ------------------------------------------------------------
# DISPLAY FINAL ANSWER
# ------------------------------------------------------------

print("\n========== FINAL ANSWER ==========\n")


# Print Llama's generated answer.
print(
    result["answer"]
)


# ------------------------------------------------------------
# DISPLAY RETRIEVED FAQs
# ------------------------------------------------------------

print("\n========== RETRIEVED FAQs ==========\n")


# Go through every FAQ retrieved by FAISS.
for faq in result["sources"]:

    # Print the FAQ question.
    print(
        "Question:",
        faq["question"]
    )

    # Print the FAQ answer.
    print(
        "Answer:",
        faq["answer"]
    )

    # Print the FAQ category.
    print(
        "Category:",
        faq["category"]
    )

    # Print similarity score.
    print(
        "Similarity:",
        faq["score"]
    )

    # Print separator.
    print("------------------------------------")