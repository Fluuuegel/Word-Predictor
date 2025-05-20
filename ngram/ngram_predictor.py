import string
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import json_formatter as formatter

def load_and_preprocess_json_files(file_paths, max_chars_per_doc=100000):
    # Load raw texts from JSON files, extract up to max_chars_per_doc characters per document,
    # remove punctuation, and lowercase everything.

    raw_texts = []
    for idx, path in enumerate(file_paths):
        docs = formatter.extract_text_from_json(path, max_chars_per_doc)
        raw_texts.extend(docs)
        print(f"Loaded shard #{idx+1}: {len(docs)} documents")

    # Precompute translation table once
    punct_translator = str.maketrans('', '', string.punctuation)
    cleaned = [text.translate(punct_translator).lower() for text in raw_texts]
    return cleaned


def train_ngram_models(texts, max_n):
    # Return a dict mapping each order n (1 to max_n) to a model of type:
    # Dict[Tuple[str, ...], Counter], where each key is a context tuple
    # (length n-1) and the value is a Counter of next word frequencies.

    models = {n: defaultdict(Counter) for n in range(1, max_n+1)}
    counter = 0
    for text in texts:
        words = text.split()
        for n in range(1, max_n+1):
            padded = ['<s>'] * (n - 1) + words + ['</s>']
            for i in range(len(words) + 1):
                ctx = tuple(padded[i : i + n - 1])
                target = padded[i + n - 1]
                models[n][ctx][target] += 1
        if (counter + 1) % 2000 == 0:
            print(f"Processed {counter + 1} texts...")
        counter += 1
    return models


def predict_next_word(models, prompt, n):
    # Return the most likely next word predicted from the context, or an empty string.

    tokens = prompt.strip().split()

    for k in range(n, 0, -1):
        model_k = models[k]
        if k == 1:
            # unigram: context is empty tuple
            counter = model_k[()]  
        else:
            # fetch the last k-1 token
            if len(tokens) >= k-1:
                ctx = tuple(tokens[-(k-1):])
            else:
                ctx = tuple(['<s>']*(k-1-len(tokens)) + tokens)
            counter = model_k.get(ctx, None)
        if counter and len(counter) > 0:
            return counter.most_common(1)[0][0]
    return ""


def evaluate(models: Dict[int, Dict[Tuple[str, ...], Counter]],
             test_texts: List[str],
             n_values: List[int]) -> Dict[int, float]:
    # Compare the saved keystrokes proportion of various N-gram models on the same set of test texts.
    # Returns: Dict[n, proportion_saved]

    saved = {n: 0 for n in n_values}
    total = {n: 0 for n in n_values}

    # Cache: (order, context_tuple) -> predicted_word or None
    best_cache: Dict[Tuple[int, Tuple[str, ...]], str] = {}

    def predict_from_tokens(tokens: List[str], idx: int, order: int) -> str:
        for k in range(order, 0, -1):
            if k == 1:
                ctx = ()
            else:
                need = k - 1
                if idx >= need:
                    ctx = tuple(tokens[idx - need : idx])
                else:
                    ctx = tuple(['<s>'] * (need - idx) + tokens[0:idx])

            key = (k, ctx)
            if key in best_cache:
                pred = best_cache[key]
            else:
                counter = models[k].get(ctx)
                pred = counter.most_common(1)[0][0] if counter else None
                best_cache[key] = pred

            if pred is not None:
                return pred
        return ""

    for text in test_texts:
        tokens = text.split()
        for idx, actual in enumerate(tokens):
            for n in n_values:
                total[n] += len(actual)
                pred = predict_from_tokens(tokens, idx, n)
                if pred == actual:
                    saved[n] += len(actual) - 1

    return {n: (saved[n] / total[n] if total[n] > 0 else 0.0)
            for n in n_values}


if __name__ == '__main__':

    shards = [f"../data/c4-train.000{i//10}{i%10}-of-01024.json" for i in range(1)]
    texts = load_and_preprocess_json_files(shards, max_chars_per_doc=10000)

    split = int(len(texts) * 0.8)
    train_texts = texts[:split]
    test_texts = texts[split:]

    max_order = 3
    models = train_ngram_models(train_texts, max_order)

    while True:
        prompt = input("Input your prompt (or 'exit'): ")
        if prompt.lower() == 'exit':
            break
        
        next_word = predict_next_word(models, prompt, n=3)
        print(f"Predicted next word: '{next_word}'")

    # Evaluate
    results = evaluate(models, test_texts, n_values=[1, 2, 3])
    for n, prop in results.items():
        print(f"{n}-gram saved proportion: {prop:.2%}")
