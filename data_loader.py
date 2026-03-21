import os
import requests
import logging

logger = logging.getLogger(__name__)

SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")


def fetch_products():
    try:
        res = requests.get(f"{SPRING_BOOT_BASE_URL}/api/products/all", timeout=5)
        if res.status_code == 200:
            data = res.json()
            products = []
            for p in data:
                products.append({
                    "id":          p.get("id"),
                    "name":        p.get("name", ""),
                    "category":    p.get("category", {}).get("name", "General") if isinstance(p.get("category"), dict) else str(p.get("category", "General")),
                    "price":       float(p.get("price", 0)),
                    "description": p.get("description", p.get("name", "")),
                })
            if products:
                logger.info(f"Loaded {len(products)} products from Spring Boot")
                return products
    except Exception as e:
        logger.warning(f"Could not fetch products from Spring Boot: {e}")
    return None


def fetch_orders():
    try:
        res = requests.get(f"{SPRING_BOOT_BASE_URL}/api/orders/all", timeout=5)
        if res.status_code == 200:
            data = res.json()
            orders = []
            for o in data:
                user_id = o.get("userId") or o.get("user", {}).get("id")
                for item in o.get("orderItems", []):
                    product_id = item.get("productId") or item.get("product", {}).get("id")
                    if user_id and product_id:
                        orders.append({"userId": user_id, "productId": product_id})
            if orders:
                logger.info(f"Loaded {len(orders)} order items from Spring Boot")
                return orders
    except Exception as e:
        logger.warning(f"Could not fetch orders from Spring Boot: {e}")
    return None


def fetch_reviews():
    try:
        res = requests.get(f"{SPRING_BOOT_BASE_URL}/api/reviews/all", timeout=5)
        if res.status_code == 200:
            data = res.json()
            reviews = []
            for r in data:
                user_id = r.get("userId") or r.get("user", {}).get("id")
                product_id = r.get("productId") or r.get("product", {}).get("id")
                rating = r.get("rating", 3)
                if user_id and product_id:
                    reviews.append({"userId": user_id, "productId": product_id, "rating": rating})
            if reviews:
                logger.info(f"Loaded {len(reviews)} reviews from Spring Boot")
                return reviews
    except Exception as e:
        logger.warning(f"Could not fetch reviews from Spring Boot: {e}")
    return None


def load_all_data(recommender):
    """
    Try to load real data from Spring Boot.
    Falls back to dummy data if Spring Boot is unreachable or DB is empty.
    """
    products = fetch_products()
    orders   = fetch_orders()
    reviews  = fetch_reviews()

    if products and len(products) >= 3:
        logger.info("Using REAL data from Spring Boot database")
        recommender.load_data(
            products,
            orders  or [],
            reviews or [],
        )
    else:
        logger.info("Using DUMMY data (Spring Boot unreachable or empty DB)")
        recommender.load_dummy_data()
