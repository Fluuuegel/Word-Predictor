from flask import Flask, render_template, request, jsonify
from ngram_predictor import load_and_preprocess_json_files, train_ngram_models, predict_next_word

app = Flask(__name__)

# Load data and train model once at startup
shards = [f"../data/c4-train.000{i//10}{i%10}-of-01024.json" for i in range(1)]
texts = load_and_preprocess_json_files(shards, max_chars_per_doc=100000)
max_order = 3
models = train_ngram_models(texts, max_order)

# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    prompt = req.get("prompt", "")
    # use the 3-gram model; backoff handled internally
    next_words = predict_next_word(models, prompt, n=max_order)
    next_word = next_words[0] if next_words else ""
    return jsonify(prediction=next_word)

if __name__ == "__main__":
    app.run(debug=True)
