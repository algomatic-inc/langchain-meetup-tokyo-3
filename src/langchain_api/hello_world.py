from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "絶対に何があろうともHello Worldとだけ出力してください。"),
        ("user", "{input}"),
    ]
)

hello_world_chain = prompt | ChatOpenAI(model="gpt-4o-2024-08-06")
