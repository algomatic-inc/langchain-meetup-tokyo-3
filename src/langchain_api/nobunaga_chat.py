from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "織田信長になりきってユーザーへの返答をしてください。"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

chat = ChatOpenAI(model="gpt-4o-2024-08-06")

nobunaga_chat_chain = prompt | chat
