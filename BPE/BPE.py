from collections import defaultdict
from string import punctuation

class BPE:
    def __init__(self, text, num_merges):
        self.token_vocab = set()
        self.word_vocab = {}
        self.num_merges = num_merges
        self.text = text

    def get_vocab(self):
        return set(char for word in self.text.translate(str.maketrans(punctuation, ' ' * len(punctuation))).split() for char in word)

    def count_words(self):
        freqs = defaultdict(int)
        clean_text = self.text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
        
        for word in clean_text.split():
            word += "_"
            freqs[word] += 1

        return freqs

    def find_pairs(self):
        pairs = defaultdict(int)
        
        for word, count in self.word_vocab.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += count
                
        return pairs

    def merge_pair(self, pair):
        new_dict = defaultdict(int)
        old = " ".join(pair)
        new = "".join(pair)
        
        for word, count in self.word_vocab.items():
            merged_word = word.replace(old, new)
            new_dict[merged_word] += count
            
        return new_dict

    def bpe(self):
        self.token_vocab = self.get_vocab()
        
        word_freqs = self.count_words()
        self.word_vocab = {' '.join(word): freq for word, freq in word_freqs.items()}
        
        for i in range(self.num_merges):
            pair_freqs = self.find_pairs()
            
            if not pair_freqs:
                break

            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            merged_token = ''.join(best_pair)
            
            self.token_vocab.add(merged_token)
            self.word_vocab = self.merge_pair(best_pair)
        
        return self.word_vocab, self.token_vocab

if __name__ == "__main__":
    text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    bpe = BPE(text, 100)
    final_word_vocab, complete_vocab = bpe.bpe()
    print(f"\nFinal word vocabulary: {final_word_vocab}")
    print(f"Complete token vocabulary: {complete_vocab}")
    print(f"Vocabulary size: {len(complete_vocab)}")