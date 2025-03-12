from langchain import OpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm=llm)

load_dotenv()

st.header('Research Tool')


# Static Prompt
# user_input = st.text_input('Enter your Prompt:')

# if st.button('Summarize'):
#     result = model.invoke(user_input)
#     st.write('Some random text')

## Dynamic Prompt

paper_input = st.selectbox("Select Research Paper Name",["Select...","Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Long (detailed explanation)"])

template = load_prompt('template.json')

# Fill the Placeholder
prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)