from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()       # loading environment variables from .env file

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    max_tokens = 50
)

prompt = "Explain the theory of relativity in simple terms."

response = model.invoke(prompt)

print("Response:", response.content)

