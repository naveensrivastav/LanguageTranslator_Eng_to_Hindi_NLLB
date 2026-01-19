import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/nllb-200-distilled-600M"
# model_name = 'google/mt5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# message  = {role: 'user/ai', 
#             content "prompt/response"}

st.header( 'English -Hindi Translator' )

if "messages" not in st.session_state :
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
    # print(f'Role: {message['role'] & }')   


userInput = st.chat_input(placeholder= 'Translate in Hindi', max_chars = 200)

if userInput:
    with st.chat_message(name= 'human'):
        st.markdown(userInput)
    st.session_state.messages.append({"role": 'human', "content" :userInput})
    
    inputs = tokenizer(
        userInput,
        return_tensors="pt"
        # return_tensors="tf" ##  Depricated Now 
        )

    translated_tokens = model.generate(
        **inputs,
        # forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"],
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(["hin_Deva"]),
        max_length=1000
    )


    with st.chat_message(name = 'ai'):
        response = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        st.markdown(response)
        st.session_state.messages.append({"role": 'ai', "content" :response})
