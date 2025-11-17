from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    "meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def run_agent(prompt):
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4
    )
    # Extract the generated text
    return response.choices[0].message["content"]
