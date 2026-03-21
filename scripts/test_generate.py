from app.pipeline.generate_pipeline import generate_answer
from app.config import OLLAMA_CLOUD_API_KEY

query = "Who invented the internet?"

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

result = generate_answer(query, llm_config)

print("\nQUESTION:\n")
print(query)

print("\nANSWER:\n")
print(result["answer"])

print("\nSOURCES:\n")
for s in result["sources"]:
    print(f"{s['section']} (page {s['page']})")

# to run: python3 -m scripts.test_generate