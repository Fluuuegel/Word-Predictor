# Word Predictor

> DD2417 Project

## N-gram Predictor

### Data

[AllenAI&#39;s c4 dataset on HuggingFace](https://huggingface.co/datasets/allenai/c4/tree/main)

### Methodologies

The N-gram algorithm predicts the next word based on the previous *n* words in a sentence.

Our `train_ngram_models()` function builds N-gram models of different orders (from unigrams up to `max_n`). For each document, it counts how often a word appears after a given context of `n-1` previous words. These counts are stored in a dictionary: for each N, we store a mapping from context tuples to `Counter` , which is the next-word frequencies.

Our `predict_next_word()` function takes a prompt and uses a **backoff strategy** to predict the next word. It tries to match the highest-order context first (e.g. trigram), and if that context isn’t found, it falls back to lower-order models until it reaches the unigram model. We return the most frequent next word in the matching context.

The `evaluate()` function measures the model's performance by estimating  **keystroke savings** —how many characters a user could avoid typing if the model correctly predicted the next word. For every word in the test set, it compares the model's top prediction with the actual word. If they match, it assumes the user only typed the first letter (saving the rest). The final output is the proportion of saved keystrokes for each model order (1-gram, 2-gram, etc.).

### How to Deploy

1. `git clone ...`
2. Download the dataset and arrange the folder as `.\data\c4-train.00000-of-01024.json`
3. Create a virtual environment: `python -m venv env`
4. ```
   $ env\Scripts\activate  # Windows
   $ . env/bin/activate  # Linux or MacOS
   ```
5. `pip install flask torch`
6. `python .\ngram\app.py`
7. Check `http://127.0.0.1:5000/`
