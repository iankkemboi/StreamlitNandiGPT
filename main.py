import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch


@st.cache_resource
def load_model():
    model_name = "kembzzz/nandi-gpt_model"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_text(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


st.title("Nandi Translator")

st.write("This app translates from English to Nandi")

model, tokenizer = load_model()

input_text = st.text_area("Enter English text to translate:", height=150)

if st.button("Translate"):
    if input_text:
        try:
            with st.spinner("Translating..."):
                translation = translate_text(model, tokenizer, input_text)
            st.subheader("Translation:")
            st.write(translation)
        except Exception as e:
            st.error(f"An error occurred during translation: {str(e)}")
    else:
        st.warning("Please enter some text to translate.")

st.markdown("---")
st.write("Model powered by Hugging Face and Streamlit")