import streamlit as st
import os
from faq_gsb import faq_in, faq_out
from langchain import HuggingFaceHub, PromptTemplate
import spacy
from sklearn.metrics.pairwise import cosine_similarity


# Set the Hugging Face Hub API token as an environment variable
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

model_id = "gpt2-medium"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
    repo_id=model_id,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 150}
)

template = """
{query}
"""

# Define your FAQ entries and responses
faq_entries = faq_in

faq_responses = faq_out


# Create a dictionary to store FAQ entries and responses
faq_dict = dict(zip(faq_entries, faq_responses))


# Load spaCy with word vectors (make sure to install and download the appropriate model)
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# Define a function to generate a response using HuggingFaceHub
def generate_response(user_input):
    template = "{query}"  # Simplify the template

    # Generate a response using HuggingFaceHub
    response = conv_model.predict(template.format(query=user_input))

    return response


# Create a Streamlit web app
def main():
    st.set_page_config(page_title="GSB", page_icon="Pic.png")
    st.image("Picture1.png", width=150,)
    st.title("Laila, the GSB's AI Virtual Assistant")
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Get user input
    user_input = st.text_input("Ask a question:")

    if st.button("Submit"):
        if user_input:
            user_input = user_input.lower()  # Convert user input to lowercase for easier matching
            # Calculate the cosine similarity between user input and FAQ entries
            similarities = []
            for entry in faq_entries:
                entry_doc = nlp(entry.lower())
                user_doc = nlp(user_input)
                similarity = cosine_similarity(
                    [entry_doc.vector],
                    [user_doc.vector]
                )[0][0]
                similarities.append(similarity)

            # Find the FAQ entry with the highest similarity
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)

            # Define a threshold for considering a match
            threshold = 0.5  # Adjust as needed

            if max_similarity >= threshold:
                # If the similarity is above the threshold, provide the corresponding FAQ response
                st.text_area("Laila: ", value=faq_responses[max_index], height=300)
            else:
                # If no match is found, generate a response using HuggingFaceHub
                response = generate_response(user_input)
                st.text_area("Laila_AI: ", value=response, height=300)


if __name__ == "__main__":
    main()
