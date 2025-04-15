from dotenv import load_dotenv
from retriever import detect_language

# Load environment variables
load_dotenv()

# Test cases
test_texts = [
    ("Hello, how are you?", "English text"),
    ("مرحبا كيف حالك؟", "Arabic text"),
    ("This is a longer English text about artificial intelligence.", "Long English text"),
    ("هذا نص طويل باللغة العربية عن الذكاء الاصطناعي.", "Long Arabic text"),
]

print("🔍 Testing Azure Language Detection\n")

for text, description in test_texts:
    print(f"Testing {description}:")
    print(f"Text: {text}")
    detected = detect_language(text)
    print(f"Detected Language: {detected}\n") 