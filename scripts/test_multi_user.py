from app.pipeline.ingest_pipeline import ingest_document
from app.pipeline.retrieve_pipeline import retrieve
from app.vectorstore.qdrant_store import create_collection

# ---- TEST USERS ----
USER_A = "user_a"
USER_B = "user_b"

DOC_A = "doc_a"
DOC_B = "doc_b"

# ---- RESET COLLECTION ----
print("\nResetting collection...\n")
create_collection()

# ---- INGEST DOCUMENTS ----
print("\nIngesting documents...\n")

ingest_document("data/test_pipeline.pdf", DOC_A, USER_A)
ingest_document("data/blackhole.pdf", DOC_B, USER_B)


def print_result(label, chunks):
    print(f"{label}: {len(chunks)} chunks")
    for c in chunks:
        print(f"  - doc_id={c['doc_id']} section={c.get('section')} page={c.get('page')}")

# ---- TEST 1: USER ISOLATION ----
print("\nTEST 1: USER ISOLATION\n")

query = "Who designed the Analytical Engine?"

print("User A asking...")
res_a = retrieve(query, user_id=USER_A)
print_result("User A", res_a)
assert res_a, "User A should retrieve chunks for Analytical Engine"
assert all(c["doc_id"] == DOC_A for c in res_a), "User A should only see doc_a"

print("\nUser B asking...")
res_b = retrieve(query, user_id=USER_B)
print_result("User B", res_b)
assert all(c["doc_id"] == DOC_B for c in res_b), "User B should only see doc_b"

# EXPECTATION:
# User A should only get their own chunks
# User B should never get User A chunks


# ---- TEST 2: DIFFERENT DATA ----
print("\nTEST 2: DIFFERENT DOCUMENTS\n")

query_b = "What are black holes?"

print("User A asking...")
res_a = retrieve(query_b, user_id=USER_A)
print_result("User A", res_a)
assert all(c["doc_id"] == DOC_A for c in res_a), "User A should only retrieve doc_a chunks"

print("\nUser B asking...")
res_b = retrieve(query_b, user_id=USER_B)
print_result("User B", res_b)
assert res_b, "User B should retrieve chunks from blackhole document"
assert all(c["doc_id"] == DOC_B for c in res_b), "User B should only retrieve doc_b chunks"

# EXPECTATION:
# User A should not retrieve User B chunks
# User B should retrieve black hole content from doc_b


# ---- TEST 3: DOC FILTERING ----
print("\nTEST 3: DOC FILTERING\n")

print("User A querying only doc_a")
res = retrieve(query, user_id=USER_A, doc_id=DOC_A)
print_result("User A/doc_a", res)
assert res, "Expected chunks for correct doc filter"
assert all(c["doc_id"] == DOC_A for c in res), "Doc filter should return only doc_a"

print("\nUser A querying wrong doc")
res = retrieve(query, user_id=USER_A, doc_id="wrong_doc")
print_result("User A/wrong_doc", res)
assert not res, "Wrong doc_id should return empty results"

# EXPECTATION:
# correct doc filter returns chunks
# wrong doc filter returns empty
# to run: python3 -m scripts.test_multi_user