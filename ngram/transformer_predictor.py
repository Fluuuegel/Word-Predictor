import json_formatter as formatter
import torch
import torch.nn as nn
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from ngram_predictor import load_and_preprocess_json_files
from transformer_model import TransformerModel
from tqdm import tqdm

def encode(text):
    # Encode the text into a tensor
    tokens = word_tokenize(text)
    counter = Counter(tokens)
    vocab = {word: i for i, (word, _) in enumerate(counter.items())}
    vocab["<unk>"] = len(vocab)  # handle unknowns
    inverse_vocab = {i: word for word, i in vocab.items()}
    vocab_size = len(vocab)

    encoded = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    data = torch.tensor(encoded, dtype=torch.long)
    return vocab, inverse_vocab, vocab_size, data

def batchify(data, batch_size):
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    return data.view(batch_size, -1).t().contiguous()

def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)

    return data, target


def train(data, vocab_size, device, epochs=5):
    
    model = TransformerModel(
        vocab_size=vocab_size,
        emb_size=200,
        nhead=2,
        nhid=200,
        nlayers=2,
        dropout=0.2
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.
        for i in tqdm(range(0, data.size(0) - 1, 35), desc="Training", unit="batch"):
            input_seq, target = get_batch(data, i)
            src_mask = model.generate_square_subsequent_mask(input_seq.size(0)).to(device)
            optimizer.zero_grad()
            output = model(input_seq.to(device), src_mask)
            loss = loss_fn(output.view(-1, vocab_size), target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.2f}")

    return model

def predict_next(model, text, vocab, inv_vocab, device, n_words=5):
    model.eval()
    tokens = text.lower().split()
    input_ids = torch.tensor([vocab.get(w, vocab["<unk>"]) for w in tokens], dtype=torch.long).unsqueeze(1).to(device)
    mask = model.generate_square_subsequent_mask(input_ids.size(0)).to(device)
    with torch.no_grad():
        output = model(input_ids, mask)
    last_logits = output[-1, 0, :]  # last timestep, first batch index
    top_ids = torch.topk(torch.softmax(last_logits, dim=-1), k=n_words).indices
    return [inv_vocab[i.item()] for i in top_ids]


if __name__ == "__main__":
    shards = [f"../data/c4-train.000{i//10}{i%10}-of-01024.json" for i in range(1)]
    texts = load_and_preprocess_json_files(shards, max_chars_per_doc=10000)
    # print(f"Loaded {len(texts)} documents.")
    # Load and preprocess data
    vocab, inv_vocab, vocab_size, data = encode(" ".join(texts))
    data = batchify(data, 32)
    # print(f"Data shape: {data.shape}")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    # model = train(data=data, vocab_size=vocab_size, device=device)

    # Save model
    # torch.save(model.state_dict(), "transformer_model.pth")
    model = TransformerModel(
        vocab_size=vocab_size,
        emb_size=200,
        nhead=2,
        nhid=200,
        nlayers=2,
        dropout=0.2
    ).to(device)
    model.load_state_dict(torch.load("transformer_model.pth"))

    # Predict next words
    while True:
        prompt = input("Input your prompt (or 'exit'): ")
        if prompt.lower() == 'exit':
            break

        predictions = predict_next(model, prompt, vocab, inv_vocab, device)
        print(f"Predicted next words: {predictions}")
