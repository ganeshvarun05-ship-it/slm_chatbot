
from transformers import pipeline


model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading SLM model... Please wait.")

chatbot = pipeline(
    "text-generation",
    model=model_name
)

print("SLM Chatbot is ready!")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot closed.")
        break

    messages = [
        {"role": "user", "content": user_input}
    ]

    response = chatbot(
        messages,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    print("Bot:", response[0]["generated_text"][-1]["content"])
