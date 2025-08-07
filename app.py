import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 🔐 Groq API key
GROQ_API_KEY = "your-groq-api-key"

# ✅ Set up custom memory that exposes `.messages`
class CompatibleMemory(ConversationBufferMemory):
    @property
    def messages(self):
        return self.chat_memory.messages

    def add_messages(self, messages):
        for msg in messages:
            if msg.type == "human":
                self.chat_memory.add_user_message(msg.content)
            elif msg.type == "ai":
                self.chat_memory.add_ai_message(msg.content)

# ✅ Only initialize memory once using Streamlit session state
if "memory" not in st.session_state:
    st.session_state.memory = CompatibleMemory(return_messages=True)

def get_memory(session_id: str):
    return st.session_state.memory

# ✅ Groq LLM setup
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# ✅ Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful healthcare assistant. You provide general health advice and help book appointments."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ✅ Combine prompt + LLM
chain = prompt | llm

# ✅ Wrap with memory history support
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history=get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# ✅ Streamlit UI
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")
st.title("🩺 Healthcare Chatbot")
st.markdown("Ask health questions or book a medical appointment.")

# 🔄 Session ID
session_id = "streamlit-session"

# 💬 Show previous messages
for msg in st.session_state.memory.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)

# 🧾 New chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)

    # Get response
    with st.spinner("Thinking..."):
        response = conversation.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

    # Show assistant message
    st.chat_message("assistant").write(response.content)
