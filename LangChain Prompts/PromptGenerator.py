# Its is from making Reusable Dynamic Prompt.

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(template="""
You are an intelligent AI tutor.

The Topic is:
{topic}

User Level in the {topic} is:
{level}

User Request's to :
{quoestion}

Instructions:
- Understand the user's request.
- Explain according to the user's level.
- Give clear and accurate information.
- Use simple examples when useful.
- Structure the answer properly.
"""
,input_variables=['topic','level','quoestion']
)

prompt.save('template.json')