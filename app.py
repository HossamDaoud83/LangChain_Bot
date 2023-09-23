import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import spacy
import numpy as np
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
faq_entries = [
    "Who are you?",
    "Who is the college Name?",
    "Who is the college Dean?",
    "Who is your Developer?",
    "How can I contact you?",
    "Could I installments the Tuition Fees?",
    "What is the Program Study Plan?",
    "How can i register in your college?",
    # Add more FAQ entries for different topics
]

faq_responses = [
    "I'm Laila, the GSB's AI Virtual Assistant.",
    "Graduate School of Business at The Arab Academy for Science, Technology and Maritime Transport (AASTMT)",
    "Prof.Aiman Ragab, current college dean since 2016, He has a PhD in Accounting and finance from USA.",
    "Eng. Hossam Daoud, GSB E-learning Administrator.",
    """College Contacts is Only: 01111781111 or 19838
    Facebook: https://www.fb.com/MBA.AAGSB.AAST.EDU
    Website: https://aast.edu/en/colleges/gsb/alex
    Address: Arab Academy, Gamal Abdel Nasser, Miami, Alexandria, Egypt
    E-mail: student@aagsb.aast.edu
    GSB Moodle: https://gsb.aast.edu""",
    "Tuition fees cannot be paid in installments",
    "Study plans through the college website only",
    "complete the online form to enroll:https://forms.gle/5dPnX6BLHSLLwLSg7, or you can contact us on 01111781111 or 19838",
    # Add corresponding responses for each FAQ entry
]

# Create a dictionary to store FAQ entries and responses
faq_dict = dict(zip(faq_entries, faq_responses))

# Load spaCy with word vectors (make sure to install and download the appropriate model)
nlp = spacy.load("en_core_web_sm")

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['query'])
    conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    user_input = message.lower()  # Convert user input to lowercase for easier matching

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
    threshold = 0.2  # Adjust as needed

    if max_similarity >= threshold:
        # If the similarity is above the threshold, provide the corresponding response
        await cl.Message(content=faq_responses[max_index]).send()
    else:
        # If no match is found, generate a response using HuggingFaceHub
        response = await llm_chain.acall(template.format(query=message), callbacks=[cl.AsyncLangchainCallbackHandler()])
        await cl.Message(content=response["text"]).send()

if __name__ == "__main__":
    cl.run()
