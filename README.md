# 🩺 Healthcare Chatbot

This is an AI-powered healthcare chatbot built with **LangChain**, **Streamlit**, and **Groq's LLMs**. It provides general health advice and appointment booking assistance, with conversational memory support.

---

## 🚀 Features

- Conversational healthcare assistant powered by **Llama3-70B** via Groq.
- Session-based memory using `ConversationBufferMemory`.
- Web UI via **Streamlit**.
- Mock testing using **unittest** and `RunnableLambda`.

---

## 🧠 Architecture Overview

- **LLM**: Groq's `llama3-70b-8192` model via `langchain_groq`.
- **Prompting**: Uses `ChatPromptTemplate` with memory placeholders.
- **Memory**: Compatible with `RunnableWithMessageHistory` using a custom memory class.
- **Frontend**: Streamlit-based interactive chat UI.
- **Tests**: Unit tests use a mock LLM that echoes user input.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/healthcare-chatbot.git
cd healthcare-chatbot
```
2. Install Dependencies
Make sure you are using Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

```bash
streamlit
langchain
langchain_groq
```

3. Get Your Groq API Key
Create an account or log in to Groq.

Generate an API key from the API Keys section.

Then, either:

Replace the placeholder in app.py:

GROQ_API_KEY = "your-groq-api-key"

Set it as an environment variable:

```bash
export GROQ_API_KEY=your-groq-api-key
```

In app.py, use:
```bash
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

▶️ Run the App
```bash
streamlit run app.py
```
This launches the chatbot UI in your browser. You can type health-related queries and receive intelligent responses with context-aware memory.

✅ Run Tests

The test suite is located in test_chatbot.py.

To run:

```bash
python -m unittest test_chatbot.py
```

Tests include:

Single response verification.

Session memory isolation checks.

📁 File Structure
```bash
├── app.py               # Streamlit frontend with memory and Groq LLM
├── test_chatbot.py      # Unit tests using mock LLM
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```
