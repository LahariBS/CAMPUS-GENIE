// =========================================================
// GET HTML ELEMENTS
// =========================================================

// Get the chat message container.
const chatMessages =
    document.getElementById("chatMessages");


// Get the question input.
const questionInput =
    document.getElementById("questionInput");


// Get the chat form.
const chatForm =
    document.getElementById("chatForm");


// Get the send button.
const sendButton =
    document.getElementById("sendButton");


// =========================================================
// BACKEND URL
// =========================================================

// This is the address where FastAPI is running.
const API_URL =
    "https://campus-genie-l2ym.onrender.com/ask";


// =========================================================
// ADD MESSAGE TO CHAT
// =========================================================

function addMessage(
    text,
    sender
) {

    // Create a new div for the message.
    const messageDiv =
        document.createElement("div");


    // Add the message class.
    messageDiv.classList.add(
        "message"
    );


    // Add user or bot class.
    messageDiv.classList.add(
        sender === "user"
            ? "user-message"
            : "bot-message"
    );


    // Create the message bubble.
    const bubble =
        document.createElement("div");


    // Add bubble class.
    bubble.classList.add(
        "message-bubble"
    );


    // Add text inside the bubble.
    bubble.textContent = text;


    // Put bubble inside message.
    messageDiv.appendChild(
        bubble
    );


    // Put message inside chat area.
    chatMessages.appendChild(
        messageDiv
    );


    // Automatically scroll to bottom.
    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


// =========================================================
// SEND QUESTION TO BACKEND
// =========================================================

chatForm.addEventListener(
    "submit",
    async function(event) {

        // Prevent page reload.
        event.preventDefault();


        // Get user's question.
        const question =
            questionInput.value.trim();


        // Do nothing if question is empty.
        if (!question) {
            return;
        }


        // Display user's question.
        addMessage(
            question,
            "user"
        );


        // Clear input.
        questionInput.value = "";


        // Disable button while waiting.
        sendButton.disabled = true;


        // Change button text.
        sendButton.textContent =
            "Thinking...";


        try {

            // Send request to FastAPI.
            const response =
                await fetch(
                    API_URL,
                    {

                        // HTTP method.
                        method: "POST",

                        // Tell backend we are sending JSON.
                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        // Convert JavaScript object to JSON.
                        body: JSON.stringify({
                            question: question
                        })
                    }
                );


            // Convert backend response into JSON.
            const data =
                await response.json();


            // Display AI answer.
            addMessage(
                data.answer,
                "bot"
            );


        } catch (error) {

            // Display error if backend is unavailable.
            addMessage(
                "Sorry, I could not connect to the server. Please make sure the FastAPI backend is running.",
                "bot"
            );


            // Print error in browser console.
            console.error(error);

        }


        // Enable button again.
        sendButton.disabled = false;


        // Restore button text.
        sendButton.textContent =
            "Send";
    }
);