from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()


model = ChatHuggingFace(
  llm = HuggingFaceEndpoint(
      model='XiaomiMiMo/MiMo-V2.6-Pro-RL',
      huggingfacehub_api_token=os.getenv('HF_API'),
      temperature=0.5
  )
)

prompt1 = PromptTemplate(
    template='Generate 5 short question answer on thee following report \n {report}',
    input_variables=['report']
)

prompt2 = PromptTemplate(
    template='Generate short and simple notes on the following report \n {report}',
    input_variables=['report']
)

prompt3 = PromptTemplate(
    template="""
You are an educational content editor.

Using the notes and quiz provided below, create one clean and
well-structured study document.

The final document MUST contain these sections:

# AI Study Guide

## 1. Summary
Give a concise summary of the main concepts.

## 2. Key Notes
Organize the important concepts using headings and bullet points.
Keep the explanations simple and beginner-friendly.

## 3. Quiz
Create exactly 5 question-answer pairs based ONLY on the provided
report and notes.

Format them as:

Q1. Question
Answer: ...

Q2. Question
Answer: ...

Do not add information that is not present in the original report.

### Notes:
{notes}

### Quiz:
{quiz}

Now produce the final study guide.
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'quiz' : prompt1 | model | parser ,
    'notes' : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain


with open('report.txt' , mode='r') as file :
  report = file.read()


result = chain.invoke({'report' : report})

print(result)

chain.get_graph().print_ascii()