from app.ingestion.loader import load_document

doc = load_document("data/sample.pdf")

for d in doc[:5]:
    print(d)