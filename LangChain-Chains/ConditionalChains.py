from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnableBranch 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

load_dotenv()


model = ChatHuggingFace(
  llm = HuggingFaceEndpoint(
      model='XiaomiMiMo/MiMo-V2.6-Pro-RL',
      huggingfacehub_api_token=os.getenv('HF_API'),
      temperature=0
  )
)


class FeedBack(BaseModel):
  sentiment : Literal['Positive', 'Negative', 'Neutral'] = Field(
    description='Sentiment of the customer feedback'
  )

pydantic_parser = PydanticOutputParser(pydantic_object=FeedBack)

parser = StrOutputParser()

classification_prompt = PromptTemplate(
    template="""
Classify the sentiment of the following customer feedback.
\n
{format_instruction}

Do not provide explanations.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
    partial_variables={'format_instruction' : pydantic_parser.get_format_instructions()}
)


Positive_feedback = '''
The customer support was excellent. The chatbot quickly understood my issue,
provided the correct solution, and helped me resolve the problem within a
few minutes. Overall, I am very satisfied with the service.
'''

Negative_feedback = '''
The customer support was very disappointing. The chatbot failed to understand
my problem and kept giving irrelevant answers. I had to contact support
multiple times before getting any help.
'''

Neutral_feedback = '''
The customer support answered my questions and provided information about my
order. The chatbot was easy to use, although the interaction was fairly
standard and nothing particularly stood out.
'''

classifier_chain = classification_prompt | model | pydantic_parser

prompt2 = PromptTemplate(
    template="""
Write a polite and appreciative response to this positive customer feedback:

{feedback}
""",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""
Write a polite and helpful response to this negative customer feedback.
Acknowledge the customer's problem and apologize where appropriate.

{feedback}
""",
    input_variables=["feedback"]
)

prompt4 = PromptTemplate(
    template="""
Write a polite and informative response to this neutral customer feedback.

{feedback}
""",
    input_variables=["feedback"]
)


branch_chain = RunnableBranch(
  (lambda x:x.sentiment == 'Positive' , prompt2 | model | parser),
  (lambda x:x.sentiment == 'Negative' , prompt3 | model | parser),
  prompt4 | model | parser
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : Negative_feedback})

print(result)

chain.get_graph().print_ascii()