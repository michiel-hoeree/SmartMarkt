import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from collections import Counter
from typing import Dict, List, Set, Tuple

DEFAULT_CSV_PATH = 'Supermart-Grocery-Sales-Retail-Analytics-Dataset-Euro.csv'


class SklearnRecommender:
    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.csv_path = csv_path
        self.df = None
        self.klanten = []
        self.producten = []
        self.aankoopgeschiedenis = {}
        self.product_popularity: Counter[str] = Counter()
        self._product_index: Dict[str, dict] = {}
        
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_features = None
        self.item_features = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.id_to_user = {}
        self.id_to_item = {}
        self.model_file = 'sklearn_model.pkl'
        
    def load_csv_data(self):
        """Load and prepare data from CSV"""
        print('Loading CSV data...')
        self.df = pd.read_csv(self.csv_path)
        
        # Create Product column
        self.df['Product'] = self.df['Category'] + ' - ' + self.df['Sub Category']
        
        # Create unique customers list with IDs
        unique_customers = self.df['Customer Name'].dropna().unique()
        self.klanten = [
            {
                'id': f'customer_{i}',
                'naam': name,
                'voorkeuren': [],  # We don't have this in CSV, so empty
                'leeftijd': 30  # Default age since not in CSV
            }
            for i, name in enumerate(unique_customers)
        ]
        
        # Create customer name to ID mapping
        customer_name_to_id = {customer['naam']: customer['id'] for customer in self.klanten}
        
        # Create unique products list with IDs
        unique_products = self.df[['Category', 'Sub Category', 'Product']].drop_duplicates()
        self.producten = [
            {
                'id': f'product_{i}',
                'naam': row['Product'],
                'categorie': row['Category'],
                'subcategorie': row['Sub Category'],
                'kenmerken': [row['Category'], row['Sub Category']],
                'prijs': 10  # Default price since not directly in CSV
            }
            for i, (_, row) in enumerate(unique_products.iterrows())
        ]

        self._rebuild_indexes()
        
        # Create product name to ID mapping
        product_name_to_id = {product['naam']: product['id'] for product in self.producten}
        
        # Build purchase history: customer_id -> [[product_ids]]
        self.aankoopgeschiedenis = {}
        product_frequency = Counter()
        for customer_name, group in self.df.groupby('Customer Name'):
            customer_id = customer_name_to_id[customer_name]

            # Determine the top preferences for the customer based on order frequency
            top_preferences = (
                group['Sub Category']
                .value_counts()
                .head(3)
                .index.tolist()
            )

            customer = next(customer for customer in self.klanten if customer['id'] == customer_id)
            customer['voorkeuren'] = top_preferences

            # Each order is a list of product IDs grouped by order id to simulate baskets
            orders = []
            for _, order_group in group.groupby('Order ID'):
                product_ids = [
                    product_name_to_id[prod]
                    for prod in order_group['Product'].unique()
                    if prod in product_name_to_id
                ]
                if product_ids:
                    orders.append(product_ids)
                    product_frequency.update(product_ids)

            self.aankoopgeschiedenis[customer_id] = orders if orders else [[]]
        
        self.product_popularity = product_frequency

        print(f'Loaded {len(self.klanten)} customers and {len(self.producten)} products')
        
    def prepare_data(self):
        """Prepare data for recommendation system"""
        # Load CSV if not already loaded
        if self.df is None:
            self.load_csv_data()
        
        # Create mappings
        self.user_id_map = {user['id']: idx for idx, user in enumerate(self.klanten)}
        self.item_id_map = {product['id']: idx for idx, product in enumerate(self.producten)}
        self.id_to_user = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.id_to_item = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        # Build user-item interaction matrix
        n_users = len(self.klanten)
        n_items = len(self.producten)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for user_id, orders in self.aankoopgeschiedenis.items():
            if user_id in self.user_id_map:
                user_idx = self.user_id_map[user_id]
                for order in orders:
                    for product_id in order:
                        if product_id in self.item_id_map:
                            item_idx = self.item_id_map[product_id]
                            # use frequency counts instead of binary 0/1 — gives more signal for heavy items
                            self.user_item_matrix[user_idx, item_idx] += 1.0
        
        # Create user features (use preferences only).
        # Avoid using unique fields like customer name or a default identical age for everyone,
        # because those make users artificially orthogonal or identical.
        user_feature_texts = []
        for user in self.klanten:
            # If no explicit preferences, use an empty string so vectorizer handles it.
            prefs = user.get('voorkeuren') or []
            # repeat preferences once to give slightly more weight to preferences vs noise
            features = prefs + prefs
            user_feature_texts.append(' '.join(features))
        
        # Create item features (characteristics + category + subcategory)
        item_feature_texts = []
        for product in self.producten:
            features = product['kenmerken'] + [product['categorie'], product['subcategorie']]
            item_feature_texts.append(' '.join(features))
        
        # Vectorize features
        vectorizer_user = TfidfVectorizer(max_features=100)
        vectorizer_item = TfidfVectorizer(max_features=100)
        self.user_features = vectorizer_user.fit_transform(user_feature_texts)
        self.item_features = vectorizer_item.fit_transform(item_feature_texts)
        
        # Calculate similarities
        self.user_similarity = cosine_similarity(self.user_features)
        self.item_similarity = cosine_similarity(self.item_features)
        
        print('Data preparation complete')
        
    def train_model(self):
        """Train the recommendation model"""
        self.prepare_data()
        
        # Save model
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'user_features': self.user_features,
            'item_features': self.item_features,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'id_to_user': self.id_to_user,
            'id_to_item': self.id_to_item,
            'klanten': self.klanten,
            'producten': self.producten,
            'aankoopgeschiedenis': self.aankoopgeschiedenis,
            'product_popularity': dict(self.product_popularity),
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f'Model trained and saved to {self.model_file}')
    
    def load_model(self):
        """Load pre-trained model if available"""
        if os.path.exists(self.model_file):
            print('Loading pre-trained model...')
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
                try:
                    self.user_item_matrix = model_data['user_item_matrix']
                    self.user_similarity = model_data['user_similarity']
                    self.item_similarity = model_data['item_similarity']
                    self.user_features = model_data['user_features']
                    self.item_features = model_data['item_features']
                    self.user_id_map = model_data['user_id_map']
                    self.item_id_map = model_data['item_id_map']
                    self.id_to_user = model_data['id_to_user']
                    self.id_to_item = model_data['id_to_item']
                    self.klanten = model_data['klanten']
                    self.producten = model_data['producten']
                    self.aankoopgeschiedenis = model_data['aankoopgeschiedenis']
                    popularity = model_data.get('product_popularity')
                    if popularity:
                        self.product_popularity = Counter(popularity)
                    else:
                        self._recompute_popularity()
                    self._rebuild_indexes()
                except KeyError:
                    print('Stored model mist velden, opnieuw trainen...')
                    return False
                print('Model loaded successfully')
                return True
        return False
    
    def get_user_id_by_name(self, customer_name):
        """Get user ID by customer name"""
        for user in self.klanten:
            if user['naam'] == customer_name:
                return user['id']
        return None
    
    def get_user_based_recommendations(self, user_id, num_recommendations=5):
        """Get recommendations based on similar users"""
        if self.user_item_matrix is None:
            if not self.load_model():
                self.train_model()
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Get user's purchase history
        user_orders = self.aankoopgeschiedenis.get(user_id, [])
        bought_items = set()
        for order in user_orders:
            for product_id in order:
                if product_id in self.item_id_map:
                    bought_items.add(self.item_id_map[product_id])
        # Find similar users (exclude self and ignore very low similarities)
        user_similarities = self.user_similarity[user_idx]
        sorted_indices = np.argsort(user_similarities)[::-1]
        # Exclude self and require a minimum similarity threshold to be considered a neighbor
        min_similarity_threshold = 0.1
        similar_users = [i for i in sorted_indices if i != user_idx and user_similarities[i] > min_similarity_threshold][:5]

        # Calculate recommendation scores
        scores = np.zeros(len(self.producten))
        for similar_user_idx in similar_users:
            similarity = user_similarities[similar_user_idx]
            user_purchases = self.user_item_matrix[similar_user_idx]
            scores += similarity * user_purchases
        
        # Filter out already bought items and sort
        for item_idx in bought_items:
            scores[item_idx] = 0
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:num_recommendations]
        
        recommendations = []
        for item_idx in top_items:
            if scores[item_idx] > 0:
                product_id = self.id_to_item[item_idx]
                product = next((p for p in self.producten if p['id'] == product_id), None)
                if product:
                    recommendations.append((product, scores[item_idx]))
        
        return recommendations
    
    def get_item_based_recommendations(self, user_id, num_recommendations=5):
        """Get recommendations based on item similarity"""
        if self.user_item_matrix is None:
            if not self.load_model():
                self.train_model()
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Get user's purchase history
        user_orders = self.aankoopgeschiedenis.get(user_id, [])
        bought_items = set()
        for order in user_orders:
            for product_id in order:
                if product_id in self.item_id_map:
                    bought_items.add(self.item_id_map[product_id])
        
        # If the user hasn't bought anything, item-based recommendations can't be computed here.
        # Return empty to let hybrid logic fall back to popularity/fallback strategies.
        if not bought_items:
            return []

        # Calculate scores based on item similarity
        scores = np.zeros(len(self.producten))
        for bought_item in bought_items:
            item_similarities = self.item_similarity[bought_item]
            scores += item_similarities
        
        # Filter out already bought items
        for item_idx in bought_items:
            scores[item_idx] = 0
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:num_recommendations]
        
        recommendations = []
        for item_idx in top_items:
            if scores[item_idx] > 0:
                product_id = self.id_to_item[item_idx]
                product = next((p for p in self.producten if p['id'] == product_id), None)
                if product:
                    recommendations.append((product, scores[item_idx]))
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id, num_recommendations=5):
        """Get hybrid recommendations combining user-based and item-based"""
        user_recs = self.get_user_based_recommendations(user_id, num_recommendations * 2)
        item_recs = self.get_item_based_recommendations(user_id, num_recommendations * 2)
        
        # Normalize scores so that user/item-based and popularity are on a comparable scale
        def _normalize(recs):
            if not recs:
                return []
            max_score = max(score for _, score in recs)
            if max_score <= 0:
                return [(p, 0.0) for p, _ in recs]
            return [(p, score / max_score) for p, score in recs]

        user_recs_norm = _normalize(user_recs)
        item_recs_norm = _normalize(item_recs)

        # Combine and deduplicate with weights
        all_recs = {}
        # Aggressive fine-tune: strongly prefer user-based recommendations
        user_weight = 0.95
        item_weight = 0.05

        for product, score in user_recs_norm:
            if product['id'] not in all_recs:
                all_recs[product['id']] = (product, score * user_weight)

        for product, score in item_recs_norm:
            if product['id'] in all_recs:
                existing_score = all_recs[product['id']][1]
                all_recs[product['id']] = (product, existing_score + score * item_weight)
            else:
                all_recs[product['id']] = (product, score * item_weight)
        
        # Sort by combined score and ensure minimum items
        sorted_recs = sorted(all_recs.values(), key=lambda x: x[1], reverse=True)

        needed = num_recommendations - len(sorted_recs)
        if needed > 0:
            existing_ids: Set[str] = {product['id'] for product, _ in sorted_recs}
            fallback = self._get_popularity_fallback(user_id, existing_ids, needed)
            # Reduce influence of fallback (popularity) by scaling their scores down (aggressive)
            fallback_scaled = [(p, score * 0.1) for p, score in fallback]
            sorted_recs.extend(fallback_scaled)

        # Enforce simple category diversity: maximaal 1 item per categorie in de top-lijst
        final_recs = []
        seen_categories = set()
        # Also limit how many strictly-popular (global top) items can appear
        top_popular_ids = {pid for pid, _ in self.product_popularity.most_common(3)}
        max_popular_in_top = 0
        popular_count = 0
        for product, score in sorted_recs:
            cat = product.get('categorie')
            # if it's a very globally popular product, limit how many we allow
            if product.get('id') in top_popular_ids:
                if popular_count >= max_popular_in_top:
                    continue
                popular_count += 1

            if cat and cat in seen_categories:
                continue
            final_recs.append((product, score))
            if cat:
                seen_categories.add(cat)
            if len(final_recs) >= num_recommendations:
                break

        # Vul aan met overige topproducten (zonder category-constraint) als niet genoeg
        if len(final_recs) < num_recommendations:
            for product, score in sorted_recs:
                if (product, score) not in final_recs:
                    final_recs.append((product, score))
                if len(final_recs) >= num_recommendations:
                    break

        return final_recs[:num_recommendations]

    def _get_popularity_fallback(self, user_id: str, exclude_ids: Set[str], limit: int) -> List[Tuple[dict, float]]:
        """Provide fallback recommendations based on global popularity."""

        if limit <= 0:
            return []

        purchased: Set[str] = set()
        for order in self.aankoopgeschiedenis.get(user_id, []):
            purchased.update(order)

        fallback_recs: List[Tuple[dict, float]] = []
        seen_ids = set(exclude_ids)

        def add_candidate(product_id: str, score: float) -> None:
            if product_id in seen_ids:
                return
            product = self._get_product_by_id(product_id)
            if not product:
                return
            fallback_recs.append((product, score))
            seen_ids.add(product_id)

        # Normalize popularity counts so they are comparable with user/item scores
        max_count = max(self.product_popularity.values()) if self.product_popularity else 1

        # 1) Personaliseer de fallback: populaire producten binnen de klantvoorkeuren
        user_prefs: Set[str] = set()
        user_obj = next((u for u in self.klanten if u['id'] == user_id), None)
        if user_obj:
            user_prefs = set(user_obj.get('voorkeuren') or [])

        # Add popular products that match the user's top subcategories first
        if user_prefs:
            for product_id, count in self.product_popularity.most_common():
                if product_id in purchased:
                    continue
                product = self._get_product_by_id(product_id)
                if not product:
                    continue
                if product.get('subcategorie') in user_prefs or product.get('categorie') in user_prefs:
                    normalized_score = float(count) / float(max_count)
                    add_candidate(product_id, normalized_score)
                    if len(fallback_recs) >= limit:
                        return fallback_recs

        # 2) Populaire producten die de klant nog niet kocht (normalized)
        for product_id, count in self.product_popularity.most_common():
            if product_id in purchased:
                continue
            normalized_score = float(count) / float(max_count)
            add_candidate(product_id, normalized_score)
            if len(fallback_recs) >= limit:
                return fallback_recs

        # 2) Populaire producten ongeacht aankoop (laag gewicht)
        for product_id, count in self.product_popularity.most_common():
            normalized_score = (float(count) / float(max_count)) * 0.5
            add_candidate(product_id, normalized_score)
            if len(fallback_recs) >= limit:
                return fallback_recs

        # 3) Als laatste redmiddel: willekeurige overige producten
        for product in self.producten:
            add_candidate(product['id'], 0.0)
            if len(fallback_recs) >= limit:
                return fallback_recs

        return fallback_recs

    def _recompute_popularity(self) -> None:
        """Recompute product popularity from purchase history."""

        popularity = Counter()
        for orders in self.aankoopgeschiedenis.values():
            for order in orders:
                popularity.update(order)

        self.product_popularity = popularity

    def _get_product_by_id(self, product_id: str) -> dict | None:
        if not self._product_index:
            self._rebuild_indexes()
        return self._product_index.get(product_id)

    def _rebuild_indexes(self) -> None:
        self._product_index = {product['id']: product for product in self.producten}


_shared_recommender: SklearnRecommender | None = None


def _get_shared_recommender(csv_path: str = DEFAULT_CSV_PATH) -> SklearnRecommender:
    global _shared_recommender

    if _shared_recommender is not None:
        return _shared_recommender

    recommender = SklearnRecommender(csv_path)
    if not recommender.load_model():
        recommender.train_model()
    else:
        if not recommender.klanten or not recommender.producten:
            recommender.load_csv_data()

    _shared_recommender = recommender
    return _shared_recommender


def train_sklearn_model(csv_path: str = DEFAULT_CSV_PATH) -> None:
    """Train and persist the sklearn recommender model."""

    global _shared_recommender

    recommender = SklearnRecommender(csv_path)
    recommender.train_model()
    _shared_recommender = None  # Force reload on next use


def get_sklearn_recommendations(
    user_id: str,
    num_recommendations: int = 5,
    csv_path: str = DEFAULT_CSV_PATH,
) -> List[Tuple[dict, float]]:
    """Return hybrid recommendations as (product, score) tuples."""

    recommender = _get_shared_recommender(csv_path)
    recommendations = recommender.get_hybrid_recommendations(user_id, num_recommendations)
    return recommendations


def get_all_sklearn_recommendations(
    num_recommendations: int = 5,
    csv_path: str = DEFAULT_CSV_PATH,
) -> Dict[str, List[Tuple[dict, float]]]:
    """Return top recommendations for every known user."""

    recommender = _get_shared_recommender(csv_path)
    results: Dict[str, List[Tuple[dict, float]]] = {}

    for user in recommender.klanten:
        user_id = user['id']
        results[user_id] = recommender.get_hybrid_recommendations(user_id, num_recommendations)

    return results


# Main execution
if __name__ == '__main__':
    # Initialize recommender
    recommender = SklearnRecommender('Supermart-Grocery-Sales-Retail-Analytics-Dataset-Euro.csv')
    
    # Train model
    print('Training model...')
    recommender.train_model()
    
    # Get recommendations for a customer (by name)
    customer_name = 'Harish'
    user_id = recommender.get_user_id_by_name(customer_name)
    
    if user_id:
        print(f'\n=== Recommendations for {customer_name} ===')
        recommendations = recommender.get_hybrid_recommendations(user_id, num_recommendations=5)
        
        if recommendations:
            for i, (product, score) in enumerate(recommendations, 1):
                print(f'{i}. {product["naam"]} (Score: {score:.4f})')
        else:
            print('No recommendations available')
    else:
        print(f'Customer {customer_name} not found')
        print('Available customers:', [k['naam'] for k in recommender.klanten[:10]])
