import streamlit as st
import os
from fuzzywuzzy import fuzz  # Import fuzzywuzzy for fuzzy string matching
from faq_gsb import faq_in, faq_out
from langchain import HuggingFaceHub
import spacy

# Set the Hugging Face Hub API token as an environment variable
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

model_id = "gpt2-large"
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


# Define a function to find the FAQ entry with the highest similarity using fuzzy string matching
def find_best_match(user_input):
    best_match_index = -1
    best_similarity = 0

    for i, entry in enumerate(faq_entries):
        similarity = fuzz.ratio(user_input, entry.lower())
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_index = i

    return best_match_index, best_similarity

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
            user_input = user_input.lower()

            # Find the best matching FAQ entry using fuzzy string matching
            best_match_index, best_similarity = find_best_match(user_input)

            # Define a threshold for considering a match (adjust as needed)
            threshold = 50  # You can change this threshold value based on your requirements

            if best_similarity >= threshold:
                # If the similarity is above the threshold, provide the corresponding FAQ response
                st.text_area("Laila: ", value=faq_responses[best_match_index], height=300)
            else:
                # If no match is found above the threshold, generate a response using HuggingFaceHub
                response = generate_response(user_input)
                st.text_area("Laila_AI: ", value=response, height=300)


if __name__ == "__main__":
    main()
