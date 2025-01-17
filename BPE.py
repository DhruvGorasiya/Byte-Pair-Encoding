from collections import defaultdict
from string import punctuation

class BPETokenizer:
    def __init__(self):
        self.word_vocab = {}
        self.token_vocab = set()

    def get_vocab(self, text):
        return set(char for word in text.translate(str.maketrans('', '', punctuation)).split() for char in word)

    def count_words(self, text):
        freqs = defaultdict(int)
        clean_text = text.translate(str.maketrans('', '', punctuation))
        
        for word in clean_text.split():
            word += "_"
            freqs[word] += 1
        
        return freqs

    def find_pairs(self, word_dict):
        pairs = defaultdict(int)
        
        for word, count in word_dict.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += count
                
        return pairs

    def merge_pair(self, word_dict, pair):
        new_dict = defaultdict(int)
        old = " ".join(pair)
        new = "".join(pair)
        
        for word, count in word_dict.items():
            merged_word = word.replace(old, new)
            new_dict[merged_word] += count
            
        return new_dict

    def train(self, text, num_merges):
        self.token_vocab = self.get_vocab(text)
        
        word_freqs = self.count_words(text)
        self.word_vocab = {' '.join(word): freq for word, freq in word_freqs.items()}
        
        for i in range(num_merges):
            pair_freqs = self.find_pairs(self.word_vocab)
            
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            merged_token = ''.join(best_pair)
            
            self.token_vocab.add(merged_token)
            self.word_vocab = self.merge_pair(self.word_vocab, best_pair)
        
        return self.word_vocab, self.token_vocab


if __name__ == "__main__":
    text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    tokenizer = BPETokenizer()
    final_word_vocab, complete_vocab = tokenizer.train(text, 3)
    print(f"\nFinal word vocabulary: {final_word_vocab}")
    print(f"Complete token vocabulary: {complete_vocab}")
    print(f"Vocabulary size: {len(complete_vocab)}")