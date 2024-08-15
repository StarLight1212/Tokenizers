from bbpe import BBPETokenizer

# Initialize the BBPE Tokenizer with the provided vocabulary and merge files
tokenizer = BBPETokenizer("vocab.json", "merges.txt")

# Encode a single string
input_text = input("Please enter the string to encode: ")
print(f"{'#' * 20} Encoding result for '{input_text}' {'#' * 20}")

encoded_ids = tokenizer.encode(input_text)
decoded_text = tokenizer.decode(encoded_ids)
tokens = tokenizer.tokenize(input_text)

print(f"Encoded IDs for '{input_text}': {encoded_ids}")
print(f"Decoded string from encoded IDs: {decoded_text}")
print(f"Tokens for '{input_text}': {tokens}")

print('#' * 60)

# Batch encoding
print("Format: list[str, ..., str], Example: ['Hello, how are you?', 'Hey, I had a great day today!']")
batch_input_text = input("Please enter the strings to encode (use the format above): ")
batch_input_text = eval(batch_input_text)  # Ensure input is evaluated as a list
batch_encoded_results = tokenizer.encode_batch(batch_input_text, num_threads=2)

print(f"{'#' * 20} Batch Encoding Results {'#' * 20}")
print(f"Input strings for batch encoding: {batch_input_text}")
print(f"Batch encoding results: {batch_encoded_results}")
