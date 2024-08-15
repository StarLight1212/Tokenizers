from bbpe import BBPETokenizer

# Load training data
with open("train_data/斗破苍穹.txt", "r", encoding="utf-8") as file:
    data = file.read()

# Define vocabulary size and output file names
vocab_size = 5000  # Size of the vocabulary
vocab_outfile = "vocab.json"  # Output file for the vocabulary
merges_outfile = "merges.txt"  # Output file for the merge rules

# Train the BBPE tokenizer on the data
BBPETokenizer.train_tokenizer(
    data,
    vocab_size,
    vocab_outfile=vocab_outfile,
    merges_outfile=merges_outfile
)
