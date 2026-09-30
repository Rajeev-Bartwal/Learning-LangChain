import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import  load_prompt


load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")


st.set_page_config(
    page_title="Dynamic AI Prompt",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Dynamic AI Prompt Generator")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    huggingfacehub_api_token=api_key,
    temperature=1.0
)

model = ChatHuggingFace(llm=llm)



if "answer" not in st.session_state:
    st.session_state.answer = None

if "prompt" not in st.session_state:
    st.session_state.prompt = None



Topic = st.text_input(
    "Input 1",
    placeholder="e.g. Python"
)

Level = st.text_input(
    "Input 2",
    placeholder="e.g. Beginner"
)

Question = st.text_area(
    "Input 3",
    placeholder="e.g. Explain Python loops with examples."
)

prompt = load_prompt('template.json')



if st.button("🚀 Generate AI Response"):

        dynamic_prompt = prompt.invoke({
            'topic':Topic,
            'level':Level,
            'quoestion':Question
        })
        
        st.session_state.prompt = dynamic_prompt

        with st.spinner("🤖 AI is thinking..."):

            try:

                response = model.invoke(dynamic_prompt)

                st.session_state.answer = response.content

            except Exception as e:

                st.error(f"Error: {e}")


if st.session_state.prompt:

    with st.expander("🔍 View Dynamic Prompt"):

        st.code(
            st.session_state.prompt,
            language="text"
        )


if st.session_state.answer:

    st.subheader("🤖 AI Response")

    st.markdown(st.session_state.answer)