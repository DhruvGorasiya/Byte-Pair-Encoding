import nltk
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
nltk.download('punkt')
nltk.download('gutenberg')

def get_vocab(text):
    # Tokenize text using NLTK's word_tokenize, keeping original case
    words = word_tokenize(text)  # Using NLTK here - this is for initial vocabulary
    vocab = Counter(words)
    return {word: freq for word, freq in vocab.items()}

def get_stats(vocab):
    # Get frequency of adjacent symbol pairs (bigrams) in vocabulary
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # Merge most frequent pair in all vocabulary words and update frequency
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def encode(text, merges):
    """
    Encode text using learned BPE merges without using NLTK
    Args:
        text (str): Text to encode
        merges (list): List of merge pairs in order of learned priority
    Returns:
        list: List of encoded tokens
    """
    # Split text into words first (using spaces and punctuation)
    words = []
    current_word = ''
    
    for char in text:
        if char.isspace() or char in '.,!?;:()[]{}""\'':
            if current_word:
                words.append(current_word)
                current_word = ''
            if not char.isspace():  # Keep punctuation as separate tokens
                words.append(char)
        else:
            current_word += char
    if current_word:  # Add the last word if exists
        words.append(current_word)
    
    # Apply BPE to each word
    encoded = []
    for word in words:
        # Split word into characters
        token = ' '.join(list(word))
        
        # Apply merges iteratively until no more can be applied
        while True:
            # Try to apply each merge rule
            old_token = token
            for pair in merges:
                bigram = ' '.join(pair)
                replacement = ''.join(pair)
                if bigram in token:
                    token = token.replace(bigram, replacement)
            
            # If no merges were applied, we're done with this word
            if old_token == token:
                break
        
        encoded.append(token.replace(' ', ''))
    
    return encoded

def decode(tokens):
    """
    Decode BPE tokens back to text
    Args:
        tokens (list): List of encoded tokens
    Returns:
        str: Decoded text
    """
    return ' '.join(tokens)

if __name__ == "__main__":
    # Training books
    book1 = gutenberg.raw("austen-emma.txt")
    book2 = gutenberg.raw("blake-poems.txt") 
    book3 = gutenberg.raw("shakespeare-hamlet.txt")
    
    # Combine training texts
    training_text = book1 + " " + book2 + " " + book3
    
    # Test books
    test_book1 = gutenberg.raw("shakespeare-caesar.txt")
    test_book2 = gutenberg.raw("carroll-alice.txt")
    test_book3 = gutenberg.raw("chesterton-ball.txt")
    
    # Create reference tokenizations using NLTK
    reference_tokenizations = {
        'shakespeare-caesar': word_tokenize(test_book1),
        'carroll-alice': word_tokenize(test_book2),
        'chesterton-ball': word_tokenize(test_book3)
    }
    
    # Initialize vocabulary with character-level splits
    vocab = get_vocab(training_text)
    vocab = {' '.join(word): freq for word, freq in vocab.items()}
    print("Initial vocabulary size:", len(vocab))

    # Number of BPE merges to perform
    num_merges = 100000

    # Train BPE and store merges
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i+1}")

    print("\nTokenization Comparison:")
    print("-" * 50)

    # Test on each book
    for book_name, ref_tokens in reference_tokenizations.items():
        test_book = gutenberg.raw(f"{book_name}.txt")
        bpe_tokens = encode(test_book, merges)
        
        # Calculate metrics
        ref_vocab = set(ref_tokens)
        bpe_vocab = set(bpe_tokens)
        
        # Basic statistics
        ref_vocab_size = len(ref_vocab)
        bpe_vocab_size = len(bpe_vocab)
        total_ref_tokens = len(ref_tokens)
        total_bpe_tokens = len(bpe_tokens)
        
        # Calculate token lengths
        ref_avg_len = sum(len(token) for token in ref_tokens) / total_ref_tokens
        bpe_avg_len = sum(len(token) for token in bpe_tokens) / total_bpe_tokens
        
        # Calculate matching metrics
        true_positives = len(ref_vocab & bpe_vocab)
        false_positives = len(bpe_vocab - ref_vocab)
        false_negatives = len(ref_vocab - bpe_vocab)
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate Jaccard similarity
        jaccard = len(ref_vocab & bpe_vocab) / len(ref_vocab | bpe_vocab) if len(ref_vocab | bpe_vocab) > 0 else 0
        
        # Calculate accuracy and coverage
        accuracy = (true_positives / total_ref_tokens) * 100 if total_ref_tokens > 0 else 0
        coverage = (len(ref_vocab & bpe_vocab) / len(ref_vocab)) * 100 if len(ref_vocab) > 0 else 0
        
        print(f"\nResults for {book_name}:")
        print(f"Reference vocabulary size: {ref_vocab_size}")
        print(f"BPE vocabulary size: {bpe_vocab_size}")
        print(f"Reference avg token length: {ref_avg_len:.2f}")
        print(f"BPE avg token length: {bpe_avg_len:.2f}")
        print(f"Total reference tokens: {total_ref_tokens}")
        print(f"Total BPE tokens: {total_bpe_tokens}")
        print(f"Tokenization accuracy: {accuracy:.2f}%")
        print(f"Tokenization coverage: {coverage:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 score: {f1_score:.2f}")
        print(f"Jaccard similarity: {jaccard:.2f}")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        print("-" * 50)