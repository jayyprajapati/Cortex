from app.pipeline.retrieve_pipeline import retrieve

query = "Which country uses wind power heavily?"

results = retrieve(query, user_id="test_user")

for r in results:
    print(r["score"], r["rerank_score"], r["section"])

# to run: python3 -m scripts.test_retrieve