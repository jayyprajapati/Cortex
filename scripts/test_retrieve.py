from app.pipeline.retrieve_pipeline import retrieve

query = "Who designed the Analytical Engine?"

results = retrieve(query)

for r in results:
    print(r["score"], r["text"][:200])

# to run: python3 -m scripts.test_retrieve