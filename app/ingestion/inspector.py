import fitz
# This module provides functions to inspect documents and extract metadata such as the number of pages, images, and text length. 
# To determine the complexity of a document, we can analyze the number of pages, images, and the length of the text. 
# This information can help us decide how to process the document for embedding and storage in a vector database.
def inspect_pdf(path):

    doc = fitz.open(path)

    image_count = 0
    text_length = 0

    for page in doc:
        image_count += len(page.get_images())
        text_length += len(page.get_text())

    return {
        "pages": len(doc),
        "images": image_count,
        "text_length": text_length
    }

def is_complex(meta):

    if meta["images"] > 2:
        return True

    if meta["pages"] > 30:
        return True

    return False