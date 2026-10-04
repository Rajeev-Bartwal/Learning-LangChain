from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# 1. Pydantic Model

class TopicInfo(BaseModel):
    title: str = Field(description="A catchy title for the topic")
    explanation: str = Field(description="Simple explanation of the topic")


# 2. Parser

parser = PydanticOutputParser(pydantic_object=TopicInfo)


# 3. LLM

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        model="Qwen/Qwen3-4B-Instruct-2507",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0
    )
)


# 4. Prompt

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
Give information about the following topic.

Topic: {topic}

{format_instructions}
""",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# 5. User Input

topic = input("Enter a topic: ")


# 6. Format Prompt

formatted_prompt = prompt.format(topic=topic)


# 7. Call LLM

response = llm.invoke(formatted_prompt)


# 8. Parse Output

result = parser.parse(response.content)


# 9. Print

print(result)