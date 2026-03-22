import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridRecommender:
    def __init__(self):
        self.products_df = None
        self.orders_df = None
        self.reviews_df = None
        self.user_item_matrix = None
        self.content_similarity = None
        self.collab_similarity = None
        self.is_trained = False

    def load_data(self, products, orders, reviews):
        """
        products: list of dicts with keys: id, name, category, price, description
        orders:   list of dicts with keys: userId, productId
        reviews:  list of dicts with keys: userId, productId, rating
        """
        self.products_df = pd.DataFrame(products)
        self.orders_df = pd.DataFrame(orders)
        self.reviews_df = pd.DataFrame(reviews)

    def load_dummy_data(self):
        products = [
            {"id": 1,  "name": "Laptop",        "category": "Electronics", "price": 999,  "description": "high performance laptop computer"},
            {"id": 2,  "name": "Mouse",          "category": "Electronics", "price": 25,   "description": "wireless mouse computer accessory"},
            {"id": 3,  "name": "Keyboard",       "category": "Electronics", "price": 45,   "description": "mechanical keyboard computer accessory"},
            {"id": 4,  "name": "Laptop Bag",     "category": "Accessories", "price": 35,   "description": "laptop bag carry travel accessory"},
            {"id": 5,  "name": "Monitor",        "category": "Electronics", "price": 299,  "description": "HD monitor display screen computer"},
            {"id": 6,  "name": "USB Hub",        "category": "Electronics", "price": 20,   "description": "usb hub computer accessory ports"},
            {"id": 7,  "name": "Webcam",         "category": "Electronics", "price": 79,   "description": "HD webcam camera video computer"},
            {"id": 8,  "name": "Headphones",     "category": "Electronics", "price": 99,   "description": "wireless headphones audio music"},
            {"id": 9,  "name": "Phone",          "category": "Electronics", "price": 699,  "description": "smartphone mobile phone device"},
            {"id": 10, "name": "Phone Case",     "category": "Accessories", "price": 15,   "description": "phone case cover protection accessory"},
            {"id": 11, "name": "Charger",        "category": "Electronics", "price": 29,   "description": "fast charger cable phone laptop"},
            {"id": 12, "name": "Desk Lamp",      "category": "Home",        "price": 40,   "description": "LED desk lamp light office home"},
            {"id": 13, "name": "Mouse Pad",      "category": "Accessories", "price": 12,   "description": "mouse pad desk accessory computer"},
            {"id": 14, "name": "Tablet",         "category": "Electronics", "price": 449,  "description": "tablet device screen portable computer"},
            {"id": 15, "name": "Speaker",        "category": "Electronics", "price": 59,   "description": "bluetooth speaker audio music portable"},
        ]

        orders = [
            {"userId": 1, "productId": 1}, {"userId": 1, "productId": 2},
            {"userId": 1, "productId": 3}, {"userId": 1, "productId": 4},
            {"userId": 2, "productId": 1}, {"userId": 2, "productId": 5},
            {"userId": 2, "productId": 6}, {"userId": 3, "productId": 2},
            {"userId": 3, "productId": 3}, {"userId": 3, "productId": 13},
            {"userId": 4, "productId": 9}, {"userId": 4, "productId": 10},
            {"userId": 4, "productId": 11}, {"userId": 5, "productId": 1},
            {"userId": 5, "productId": 7}, {"userId": 5, "productId": 8},
            {"userId": 6, "productId": 14}, {"userId": 6, "productId": 11},
            {"userId": 6, "productId": 15}, {"userId": 7, "productId": 9},
            {"userId": 7, "productId": 11}, {"userId": 7, "productId": 10},
            {"userId": 8, "productId": 1}, {"userId": 8, "productId": 2},
            {"userId": 8, "productId": 6}, {"userId": 9, "productId": 5},
            {"userId": 9, "productId": 12}, {"userId": 10, "productId": 8},
            {"userId": 10, "productId": 15},
        ]

        reviews = [
            {"userId": 1, "productId": 1, "rating": 5},
            {"userId": 1, "productId": 2, "rating": 4},
            {"userId": 2, "productId": 1, "rating": 4},
            {"userId": 2, "productId": 5, "rating": 5},
            {"userId": 3, "productId": 3, "rating": 5},
            {"userId": 4, "productId": 9, "rating": 4},
            {"userId": 4, "productId": 10, "rating": 3},
            {"userId": 5, "productId": 7, "rating": 5},
            {"userId": 6, "productId": 14, "rating": 4},
            {"userId": 7, "productId": 9, "rating": 5},
            {"userId": 8, "productId": 1, "rating": 5},
            {"userId": 9, "productId": 5, "rating": 4},
            {"userId": 10, "productId": 8, "rating": 4},
        ]

        self.load_data(products, orders, reviews)

    def train(self):
        if self.products_df is None:
            self.load_dummy_data()
        self._build_content_similarity()
        self._build_collaborative_similarity()
        self.is_trained = True

    def _build_content_similarity(self):
        # TF-IDF on product description + category
        self.products_df["combined"] = (
            self.products_df["description"] + " " + self.products_df["category"]
        )
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.products_df["combined"])

        # Normalize price and add as a feature
        scaler = StandardScaler()
        price_scaled = scaler.fit_transform(
            self.products_df[["price"]]
        )

        # Combine TF-IDF with price similarity (80/20 weight)
        text_sim = cosine_similarity(tfidf_matrix)
        price_sim = cosine_similarity(price_scaled)
        self.content_similarity = 0.8 * text_sim + 0.2 * price_sim

    def _build_collaborative_similarity(self):
        if self.orders_df.empty:
            n = len(self.products_df)
            self.collab_similarity = np.zeros((n, n))
            return

        # Build user-item matrix from orders (binary) + ratings (if available)
        all_user_ids = list(
            set(self.orders_df["userId"].tolist() +
                (self.reviews_df["userId"].tolist() if not self.reviews_df.empty else []))
        )
        all_product_ids = self.products_df["id"].tolist()

        matrix = pd.DataFrame(0.0, index=all_user_ids, columns=all_product_ids)

        # Fill purchase data (weight = 1.0)
        for _, row in self.orders_df.iterrows():
            if row["productId"] in matrix.columns:
                matrix.loc[row["userId"], row["productId"]] = 1.0

        # Add ratings on top (weight = rating/5)
        if not self.reviews_df.empty:
            for _, row in self.reviews_df.iterrows():
                if row["productId"] in matrix.columns and row["userId"] in matrix.index:
                    matrix.loc[row["userId"], row["productId"]] = max(
                        matrix.loc[row["userId"], row["productId"]],
                        row["rating"] / 5.0
                    )

        self.user_item_matrix = matrix
        item_matrix = matrix.T  # products as rows
        self.collab_similarity = cosine_similarity(item_matrix)

    def recommend(self, product_id: int, user_id: int = None, top_n: int = 5):
        if not self.is_trained:
            self.train()

        product_ids = self.products_df["id"].tolist()

        if product_id not in product_ids:
            return []

        idx = product_ids.index(product_id)

        # --- Content-based score ---
        content_scores = self.content_similarity[idx].copy()

        # --- Collaborative score ---
        collab_scores = self.collab_similarity[idx].copy()

        # --- Rating boost ---
        rating_boost = np.zeros(len(product_ids))
        if not self.reviews_df.empty:
            avg_ratings = (
                self.reviews_df.groupby("productId")["rating"].mean()
            )
            for i, pid in enumerate(product_ids):
                if pid in avg_ratings.index:
                    rating_boost[i] = avg_ratings[pid] / 5.0

        # --- User history penalty (don't re-recommend already bought) ---
        user_bought = set()
        if user_id and self.user_item_matrix is not None:
            if user_id in self.user_item_matrix.index:
                bought_row = self.user_item_matrix.loc[user_id]
                user_bought = set(
                    bought_row[bought_row > 0].index.tolist()
                )

        # --- Hybrid score (weighted combination) ---
        hybrid_scores = (
            0.5 * content_scores +
            0.35 * collab_scores +
            0.15 * rating_boost
        )

        # Build results, excluding the input product and already-bought products
        results = []
        for i, pid in enumerate(product_ids):
            if pid == product_id:
                continue
            if pid in user_bought:
                continue
            product_row = self.products_df[self.products_df["id"] == pid].iloc[0]
            results.append({
                "productId": int(pid),
                "name": product_row["name"],
                "category": product_row["category"],
                "price": float(product_row["price"]),
                "score": round(float(hybrid_scores[i]), 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    def get_popular(self, top_n: int = 5):
        """Fallback: return top rated / most purchased products"""
        if self.orders_df.empty:
            return []

        purchase_counts = self.orders_df["productId"].value_counts()
        results = []
        for pid, count in purchase_counts.head(top_n).items():
            product_row = self.products_df[self.products_df["id"] == pid]
            if product_row.empty:
                continue
            product_row = product_row.iloc[0]
            results.append({
                "productId": int(pid),
                "name": product_row["name"],
                "category": product_row["category"],
                "price": float(product_row["price"]),
                "score": round(count / len(self.orders_df), 4),
            })
        return results
