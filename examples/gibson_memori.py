from openai import OpenAI
from memori import Memori

# Initialize OpenAI client
client = OpenAI()

gibson_memory = Memori(
    database_connect="mysql+mysqlconnector://{link}",
    conscious_ingest=True,
    auto_ingest=True,
    verbose=True,
)

gibson_memory.enable()

print("Memori with Gibson - Chat with GPT-4o while memory is being used")

while 1:
    try:
        user_input = input("User: ")
        if not user_input.strip():
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print("Processing your message with memory tracking...")
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": user_input}]
        )
        print(f"AI: {response.choices[0].message.content}")
        print()  # Add blank line for readability
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue