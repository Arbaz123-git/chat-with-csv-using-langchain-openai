import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from apikey import apikey

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Define Streamlit app
def app():
    # Title and description
    st.title("CSV Query App")
    st.write("Upload a CSV file and enter a query to get an answer.")
    file = st.file_uploader("Upload CSV file", type=['csv'])
    if not file:
        st.stop()

    data = pd.read_csv(file)
    st.write('Data Preview:')
    st.dataframe(data.head())

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)

    query = st.text_input("Enter a query:")

    # Initialize conversation history
    if 'conversations' not in st.session_state:
        st.session_state['conversations'] = []

    if st.button("Execute"):
        answer = agent.run(query)

        # Append the question and answer to the conversation history
        st.session_state['conversations'].append((query, answer))

        st.write("Answer:")
        st.write(answer)

    # Display conversation history
    st.sidebar.title("Conversation History")
    for i, (question, answer) in enumerate(st.session_state['conversations']):
        st.sidebar.text(f"Q{i + 1}: {question}")
        st.sidebar.text(f"A{i + 1}: {answer}")

if __name__ == "__main__":
    app()

