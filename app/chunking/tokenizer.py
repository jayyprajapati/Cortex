# This module provides a function to count the number of tokens in a given text using the tiktoken library, 
# which is commonly used for tokenization in natural language processing tasks. 
# The token_count function encodes the input text and returns the length of the resulting token list, giving an estimate of how many tokens are present in the text. 
# This can be useful for managing input sizes when working with language models that have token limits.
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def token_count(text):
    return len(encoding.encode(text))