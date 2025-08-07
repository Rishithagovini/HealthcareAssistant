import unittest
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory


# ✅ Memory wrapper to expose `.messages` and `.add_messages`
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


class TestHealthChatbot(unittest.TestCase):
    def setUp(self):
        # Prompt template with memory placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a health assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # ✅ Mock LLM: just return a string
        def mock_llm_fn(chat_prompt_value, config=None):
            last_message = chat_prompt_value.to_messages()[-1].content
            return f"Echo: {last_message}"

        self.mock_llm = RunnableLambda(mock_llm_fn)
        self.chain = self.prompt | self.mock_llm

        # Provide session memory using CompatibleMemory
        def get_memory(session_id: str):
            return CompatibleMemory(return_messages=True)

        self.conversational_chain = RunnableWithMessageHistory(
            self.chain,
            get_session_history=get_memory,
            input_messages_key="input",
            history_messages_key="history"
        )

    def test_single_response(self):
        result = self.conversational_chain.invoke(
            {"input": "Hello"},
            config={"configurable": {"session_id": "test_session"}}
        )
        self.assertEqual(result, "Echo: Hello")

    def test_memory_isolation(self):
        result1 = self.conversational_chain.invoke(
            {"input": "Book appointment"},
            config={"configurable": {"session_id": "session_1"}}
        )
        result2 = self.conversational_chain.invoke(
            {"input": "What's my name?"},
            config={"configurable": {"session_id": "session_2"}}
        )
        self.assertEqual(result1, "Echo: Book appointment")
        self.assertEqual(result2, "Echo: What's my name?")


if __name__ == "__main__":
    unittest.main()
