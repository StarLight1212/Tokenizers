import torch
import torch.nn as nn
import regex as re
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def bytes_to_unicode():
    """
    Returns a mapping from UTF-8 bytes to Unicode strings. This avoids mapping 
    to whitespace/control characters that BBPE codes rely on. Reversible BBPE 
    codes work on Unicode strings, which requires many Unicode characters in 
    your vocabulary to avoid UNKs in a large token dataset. To avoid this, we 
    create a lookup table between UTF-8 bytes and Unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + 
        list(range(ord("¡"), ord("¬") + 1)) + 
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class BBPETokenizer(nn.Module):

    def __init__(self, vocab_path: str, merges_path: str):
        super().__init__()
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Load token merge rules
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = f.read()

        # Store merges as a list of tuples, removing the last blank line
        merges = [tuple(merge_str.split()) for merge_str in merges.split("\n")[:-1]]

        # Map tokens to BBPE decoding indices
        self.encoder = vocab
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Byte-to-Unicode character mapping for 256 characters
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Rank BBPE merges
        self.bbpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        # Pre-tokenization split regex pattern
        self.pat = re.compile(r"""
                                 's|'t|'re|'ve|'m|'ll|'d|  # Common contractions
                                 \ ?\p{L}+|\ ?\p{N}+|  # Optional space followed by 1+ Unicode letters or digits
                                 \ ?[^\s\p{L}\p{N}]+|  # Optional space followed by 1+ non-whitespace/letter/digit
                                 \s+(?!\S)|  # 1+ whitespace chars not followed by non-whitespace
                                 \s+  # 1+ whitespace chars
                                 """, re.X)

    def forward(self, text):
        if isinstance(text, list):
            # Encode in batches
            tokens = self.encode_batch(text)
            tokens = [token for row in tokens for token in row]
        else:
            # Encode a single string
            tokens = self.encode(text)
        return torch.tensor(tokens)

    def bbpe(self, token):
        '''
        Apply BBPE merge rules to a token.
        '''
        if token in self.cache:
            return self.cache[token]

        chars = [i for i in token]
        # Attempt to merge any adjacent character pairs based on the BBPE ranks
        for pair in self.bbpe_ranks.keys():
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    chars = chars[:i] + ["".join(pair)] + chars[i + 2:]
                else:
                    i += 1
        self.cache[token] = chars
        return chars

    def encode(self, text: str) -> list[int]:
        '''
        Encode a string into BBPE tokens.
        '''
        bbpe_tokens_id = []
        # Split text using the regex pattern before inputting into the BBPE algorithm
        for token in re.findall(self.pat, text):
            # Convert token to its byte representation and map to its Unicode representation
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # Apply BBPE merges and map results to their BBPE indices using the encoder
            bbpe_tokens_id.extend(self.encoder[bpe_token] for bpe_token in self.bbpe(token))
        return bbpe_tokens_id

    def tokenize(self, text):
        """
        Tokenize a string into BBPE tokens.
        :param text: The text to be tokenized.
        :return: A list of BBPE tokens.
        """
        bbpe_tokens = []
        # Split text using the regex pattern before inputting into the BBPE algorithm
        for token in re.findall(self.pat, text):
            # Convert token to its byte representation and map to its Unicode representation
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # Apply BBPE merges and collect results
            bbpe_tokens.extend(bpe_token for bpe_token in self.bbpe(token))
        return bbpe_tokens

    def encode_batch(self, batch: list[str], num_threads=4):
        '''
        Encode a list of strings into BBPE tokens in parallel.
        '''
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            result = executor.map(self.encode, batch)
        return list(result)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text

    @staticmethod
    def train_tokenizer(data, vocab_size, vocab_outfile=None, merges_outfile=None):
        """
        Train a BBPE tokenizer on a dataset.
        :param data: Training text data.
        :param vocab_size: The size of the vocabulary to retain.
        :param vocab_outfile: Filename to save the vocabulary.
        :param merges_outfile: Filename to save the byte merges.
        """

        if vocab_size < 256:
            raise ValueError("vocab_size must be greater than 256")

        # Pre-tokenize data
        byte_encoder = bytes_to_unicode()
        pat_str = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        split_words = [
            [byte_encoder[b] for b in token.encode("utf-8")] for token in re.findall(pat_str, data)
        ]
        # Add base vocabulary
        vocab = set(byte_encoder.values())
        merges = []

        # Build vocabulary until the desired size is reached
        while len(vocab) < vocab_size:
            print(len(vocab))
            pair_freq = Counter()
            # Identify the most frequent pair
            for split_word in split_words:
                pair_freq.update(zip(split_word[:-1], split_word[1:]))
            most_common_pair = pair_freq.most_common(1)[0][0]

            # Update vocabulary and merge list
            new_token = most_common_pair[0] + most_common_pair[1]
            vocab.add(new_token)
            merges.append(most_common_pair)

            # Perform the merge on the data
            new_split_words = []
            for split_word in split_words:
                i = 0
                new_word = []
                # Attempt to merge characters in the word
                while i < len(split_word) - 1:
                    if (split_word[i], split_word[i + 1]) == most_common_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(split_word[i])
                        i += 1
                if i == len(split_word) - 1:
                    new_word.append(split_word[i])
                new_split_words.append(new_word)
            split_words = new_split_words

        vocab = sorted(list(vocab))
        # Save vocabulary and merges if specified
        if merges_outfile is not None:
            with open(merges_outfile, "w", encoding="utf-8") as f:
                for merge in merges:
                    f.write(merge[0] + " " + merge[1] + "\n")
        if vocab_outfile is not None:
            with open(vocab_outfile, "w", encoding="utf-8") as f:
                json.dump({v: i for i, v in enumerate(vocab)}, f, ensure_ascii=False)
