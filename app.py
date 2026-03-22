from flask import Flask, request, jsonify
from recommender import HybridRecommender
from data_loader import load_all_data
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
recommender = HybridRecommender()


def initialize():
    logger.info("Loading data and training model...")
    load_all_data(recommender)
    recommender.train()
    logger.info("Model ready!")


# ─── Health Check ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP", "model_trained": recommender.is_trained}), 200


# ─── Recommend by productId (+ optional userId) ──────────────────────────────

@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json()
    if not body or "productId" not in body:
        return jsonify({"error": "productId is required"}), 400

    product_id = int(body["productId"])
    user_id    = body.get("userId")
    top_n      = int(body.get("topN", 5))

    results = recommender.recommend(product_id, user_id=user_id, top_n=top_n)

    if not results:
        # fallback to popular products
        results = recommender.get_popular(top_n=top_n)

    return jsonify({
        "productId":       product_id,
        "userId":          user_id,
        "recommendations": results,
        "count":           len(results),
    }), 200


# ─── Recommend by userId (personalised homepage) ─────────────────────────────

@app.route("/recommend/user/<int:user_id>", methods=["GET"])
def recommend_for_user(user_id):
    top_n = int(request.args.get("topN", 5))

    # Find the last product the user interacted with
    last_product = None
    if recommender.user_item_matrix is not None and user_id in recommender.user_item_matrix.index:
        row = recommender.user_item_matrix.loc[user_id]
        bought = row[row > 0]
        if not bought.empty:
            last_product = int(bought.index[-1])

    if last_product:
        results = recommender.recommend(last_product, user_id=user_id, top_n=top_n)
    else:
        results = recommender.get_popular(top_n=top_n)

    return jsonify({
        "userId":          user_id,
        "recommendations": results,
        "count":           len(results),
    }), 200


# ─── Popular products ─────────────────────────────────────────────────────────

@app.route("/popular", methods=["GET"])
def popular():
    top_n = int(request.args.get("topN", 5))
    results = recommender.get_popular(top_n=top_n)
    return jsonify({"recommendations": results, "count": len(results)}), 200


# ─── Retrain with fresh data ──────────────────────────────────────────────────

@app.route("/retrain", methods=["POST"])
def retrain():
    logger.info("Retraining model with fresh data...")
    load_all_data(recommender)
    recommender.train()
    return jsonify({"status": "retrained", "model_trained": recommender.is_trained}), 200


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
