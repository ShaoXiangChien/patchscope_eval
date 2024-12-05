import streamlit as st
import openai
from openai import OpenAI
import os
import json
import random
import pandas as pd
import csv
from datetime import datetime

openai.api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI()

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.jsonl'):
        # Read JSONL file line by line
        data = []
        content = uploaded_file.getvalue().decode('utf-8').strip().split('\n')
        for line in content:
            data.append(json.loads(line))
        df = pd.DataFrame(data)
    
    # Validate required columns
    required_columns = ['id', 'text']
    if not all(col in df.columns for col in required_columns):
        st.error("Uploaded file must contain 'id' and 'text' columns")
        return None
    
    return df

def save_result(result, save_format, save_path, mode='a'):
    """Save a single result to file in specified format"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_format == "JSONL":
            with open(save_path, mode, encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        else:  # CSV
            write_header = not os.path.exists(save_path)
            with open(save_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'text', 'type', 'source_id'])
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def generate_paraphrase(text, model_name, number_of_paraphrases=15):
    messages = [
        {"role": "system", "content": "You are an assistant that paraphrases sentences while preserving their meaning."},
        {"role": "user", "content": f"Generate {number_of_paraphrases} paraphrases for the following sentence:\n\n\"{text}\", and output them in JSON format like this: `['<paraphrase1>', '<paraphrase2>', '<paraphrase3>', ...]`"}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        results = response.choices[0].message.content.replace("```json", "").replace("```", "")
        results = json.loads(results)
        return list(results.values())[0]
    except Exception as e:
        st.error(f"Error generating paraphrase: {e}")
        st.error(f"Response: {results}")
        st.stop()
        return None

def generate_adversarial_example(text, model_name, number_of_adversarial_examples=10):
    messages = [
        {"role": "system", "content": "You are an assistant that creates adversarial examples to challenge language models."},
        {"role": "user", "content": f"Create {number_of_adversarial_examples} adversarial examples by negating the following sentence:\n\n\"{text}\", and output them in JSON format like this: `['output1', 'output2', 'output3', ...]`"}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1,
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        results = response.choices[0].message.content.replace("```json", "").replace("```", "")
        results = json.loads(results)
        return list(results.values())[0]
    except Exception as e:
        st.error(f"Error generating adversarial example: {e}")
        st.error(f"Response: {results}")
        print(f"Response:\n\n{results}")
        st.stop()
        return None

def generate_noisy_input(text, noise_level=0.1, number_of_noisy_inputs=5):
    noisy_inputs = []
    for _ in range(number_of_noisy_inputs):
        noisy_text = ""
        for char in text:
            if char.isalpha() and random.random() < noise_level:
                noisy_text += random.choice('abcdefghijklmnopqrstuvwxyz')
            else:
                noisy_text += char
        noisy_inputs.append(noisy_text)
    return noisy_inputs

if not openai.api_key:
    st.error("OpenAI API key is not set. Please set it in the Streamlit secrets.")

if __name__ == "__main__":
    st.title("Synthetic Data Generator")

    st.markdown("""
    This app generates synthetic data for the entire dataset.
    1. Upload a CSV or JSONL file containing 'id' and 'text' columns
    2. Select the types of synthetic data to generate
    3. Generate and save the results locally
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload Data File (CSV or JSONL)", type=['csv', 'jsonl'])
    
    if uploaded_file is None:
        st.warning("Please upload a data file")
        st.stop()

    # Load and display data
    raw_df = load_data(uploaded_file)
    if raw_df is None:
        st.stop()

    st.subheader("Uploaded Dataset")
    st.dataframe(raw_df)
    st.write(f"Total records: {len(raw_df)}")

    # choose the starting index
    start_index = st.number_input("Start index:", min_value=0, max_value=len(raw_df) - 1, value=0)
    df = raw_df.iloc[start_index:]

    # Options for synthetic data generation
    st.subheader("Select Synthetic Data Types to Generate:")
    paraphrase_option = st.checkbox("Paraphrase")
    adversarial_option = st.checkbox("Adversarial Example")
    noise_option = st.checkbox("Noisy Input")

    # Model selection
    st.subheader("Select OpenAI Model:")
    model_option = st.selectbox("Choose model:", ["gpt-3.5-turbo", "gpt-4o"])

    if not (paraphrase_option or adversarial_option or noise_option):
        st.warning("Please select at least one type of synthetic data to generate")
        st.stop()

    # Noise level slider
    if noise_option:
        noise_level = st.slider("Select Noise Level (Higher means more noise):", min_value=0.0, max_value=0.5, value=0.1)
    else:
        noise_level = 0.0

    # Save format selection before generation
    save_format = st.selectbox("Select save format:", ["JSONL", "CSV"])
    default_filename = f"synthetic_data.{save_format.lower()}"
    save_filename = st.text_input("Enter filename:", default_filename)
    
    # Generate synthetic data when the button is clicked
    if st.button("Generate Synthetic Data"):
        save_path = os.path.join('data', save_filename)
        
        # Initialize counters for summary
        total_generated = 0
        type_counts = {'paraphrase': 0, 'adversarial': 0, 'noisy': 0}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        summary_text = st.empty()

        total_rows = len(df)
        for idx, row in enumerate(df.iterrows()):
            row = row[1]  # Get the row data
            input_text = row['text']
            source_id = row['id']
            
            status_text.text(f"Processing record {idx + 1} of {total_rows}")

            try:
                if paraphrase_option:
                    paraphrases = generate_paraphrase(input_text, model_option)
                    if paraphrases:
                        for p in paraphrases:
                            result = {
                                "id": f"{source_id}_p{type_counts['paraphrase']}",
                                "text": p,
                                "type": "paraphrase",
                                "source_id": source_id
                            }
                            # Save each result immediately
                            if save_result(result, save_format, save_path):
                                type_counts['paraphrase'] += 1
                                total_generated += 1

                if adversarial_option:
                    adversarial_examples = generate_adversarial_example(input_text, model_option)
                    if adversarial_examples:
                        for a in adversarial_examples:
                            result = {
                                "id": f"{source_id}_a{type_counts['adversarial']}",
                                "text": a,
                                "type": "adversarial",
                                "source_id": source_id
                            }
                            # Save each result immediately
                            if save_result(result, save_format, save_path):
                                type_counts['adversarial'] += 1
                                total_generated += 1

                if noise_option:
                    noisy_inputs = generate_noisy_input(input_text, noise_level)
                    for n in noisy_inputs:
                        result = {
                            "id": f"{source_id}_n{type_counts['noisy']}",
                            "text": n,
                            "type": "noisy",
                            "source_id": source_id
                        }
                        # Save each result immediately
                        if save_result(result, save_format, save_path):
                            type_counts['noisy'] += 1
                            total_generated += 1

            except Exception as e:
                st.error(f"Error processing record {source_id}: {str(e)}")
                continue

            # Update progress and summary
            progress_bar.progress((idx + 1) / total_rows)
            summary_text.write(f"""
            **Current Progress:**
            - Processed: {idx + 1}/{total_rows} records
            - Total generated: {total_generated}
            - By type:
                - Paraphrase: {type_counts['paraphrase']}
                - Adversarial: {type_counts['adversarial']}
                - Noisy: {type_counts['noisy']}
            """)

        status_text.text("Processing complete!")
        st.success(f"All data has been saved to `{save_path}`")
