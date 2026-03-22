from app.pipeline.generate_pipeline import generate_answer
from app.config import OLLAMA_CLOUD_API_KEY

query = "Who invented the internet?"
# query = "Who designed the Analytical Engine?"

# for offline ollama model
# llm_config = {
#     "provider": "ollama_local",  # change this to test others
#     "model": "llama3"
# }

# for ollama cloud
llm_config = {
    "provider": "ollama_cloud",
    "api_key": OLLAMA_CLOUD_API_KEY,
    "model": "gpt-oss:120b"
}

result = generate_answer(query, llm_config, user_id="test_user")

print("\nQUESTION:\n")
print(query)

print("\nANSWER:\n")
print(result["answer"])

print("\nSOURCES:\n")

if not result["sources"]:
    print("No sources found")
else:
    for i, s in enumerate(result["sources"]):
        print(f"[Source {i+1}] {s['section']} (page {s['page']})")

# to run: python3 -m scripts.test_generate