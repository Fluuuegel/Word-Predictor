from flask import Flask, render_template, request, jsonify
from ngram_predictor import load_and_preprocess_json_files, train_ngram_models, predict_next_word
from transformer_predictor import encode, predict_next
from transformer_model import TransformerModel
import torch
import pickle

app = Flask(__name__)

# Load data and train model once at startup
# shards = [f"../data/c4-train.000{i//10}{i%10}-of-01024.json" for i in range(1)]
# texts = load_and_preprocess_json_files(shards, max_chars_per_doc=100000)
max_order = 3
# models = train_ngram_models(texts, max_order)
# save the models

# with open('models.pkl', 'wb') as f:
#     pickle.dump(models, f)

with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

shards = [f"../data/c4-train.000{i//10}{i%10}-of-01024.json" for i in range(1)]
texts = load_and_preprocess_json_files(shards, max_chars_per_doc=10000)
vocab, inv_vocab, vocab_size, data = encode(" ".join(texts))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(
    vocab_size=vocab_size,
    emb_size=200,
    nhead=2,
    nhid=200,
    nlayers=2,
    dropout=0.2
).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))

# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    prompt = req.get("prompt", "")
    # # use the 3-gram model; backoff handled internally
    # next_words = predict_next_word(models, prompt, n=max_order)
    # next_word = next_words[0] if next_words else ""
    # return jsonify(prediction=next_word)
    next_words = predict_next_word(models, prompt, n=max_order)
    next_word = next_words[0] if next_words else ""
    
    # use the transformer model; backoff handled internally
    predicted_words = predict_next(model, prompt, vocab, inv_vocab, device)
    return jsonify(prediction=next_word, predicted_words=predicted_words)

if __name__ == "__main__":
    app.run(debug=False)
