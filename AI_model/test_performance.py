import unittest
import time
from sklearn_recommend import SklearnRecommender, get_sklearn_recommendations
from data import klanten, producten, aankoopgeschiedenis


class TestPerformanceAndData(unittest.TestCase):
    
    def setUp(self):
        """Setup test data"""
        self.recommender = SklearnRecommender()
        if not self.recommender.load_model():
            self.recommender.load_csv_data()
            self.recommender.train_model()
    
    def test_data_integrity(self):
        """Test dat alle data correct is geladen"""
        self.assertGreater(len(klanten), 0, "Geen klanten geladen")
        self.assertGreater(len(producten), 0, "Geen producten geladen")
        self.assertGreater(len(aankoopgeschiedenis), 0, "Geen aankoopgeschiedenis geladen")
        
        # Check dat elke klant een id en naam heeft
        for klant in klanten:
            self.assertIn('id', klant)
            self.assertIn('naam', klant)
            self.assertIsInstance(klant['id'], str)
            self.assertIsInstance(klant['naam'], str)
        
        # Check dat elk product valide velden heeft
        for product in producten:
            self.assertIn('id', product)
            self.assertIn('naam', product)
            self.assertIn('categorie', product)
            self.assertIn('subcategorie', product)
    
    def test_recommendation_performance(self):
        """Test dat aanbevelingen snel worden gegenereerd (< 2 seconden)"""
        if not klanten:
            self.skipTest("Geen klanten geladen")
        
        user_id = klanten[0]['id']
        
        start_time = time.time()
        recommendations = get_sklearn_recommendations(user_id, 5)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        self.assertLess(execution_time, 2.0, 
                       f"Aanbevelingen duurden te lang: {execution_time:.2f}s")
        self.assertIsInstance(recommendations, list)
    
    def test_all_users_get_recommendations(self):
        """Test dat alle klanten aanbevelingen kunnen krijgen"""
        failed_users = []
        
        for klant in klanten[:10]:  # Test eerste 10 klanten
            user_id = klant['id']
            try:
                recommendations = get_sklearn_recommendations(user_id, 3)
                # Aanbevelingen mogen leeg zijn als klant niks heeft gekocht
                self.assertIsInstance(recommendations, list)
            except Exception as e:
                failed_users.append((user_id, str(e)))
        
        self.assertEqual(len(failed_users), 0, 
                        f"Aanbevelingen faalden voor: {failed_users}")
    
    def test_no_duplicate_recommendations(self):
        """Test dat er geen duplicate aanbevelingen zijn"""
        if not klanten:
            self.skipTest("Geen klanten geladen")
        
        user_id = klanten[0]['id']
        recommendations = get_sklearn_recommendations(user_id, 10)
        
        product_ids = [product['id'] for product, score in recommendations]
        unique_ids = set(product_ids)
        
        self.assertEqual(len(product_ids), len(unique_ids), 
                        "Er zijn duplicate aanbevelingen gevonden")
    
    def test_model_consistency(self):
        """Test dat het model consistente resultaten geeft"""
        if not klanten:
            self.skipTest("Geen klanten geladen")
        
        user_id = klanten[0]['id']
        
        # Krijg aanbevelingen twee keer
        recs1 = get_sklearn_recommendations(user_id, 5)
        recs2 = get_sklearn_recommendations(user_id, 5)
        
        # De volgorde moet hetzelfde zijn (determinisme)
        ids1 = [p['id'] for p, _ in recs1]
        ids2 = [p['id'] for p, _ in recs2]
        
        self.assertEqual(ids1, ids2, "Model geeft inconsistente resultaten")


if __name__ == '__main__':
    unittest.main(verbosity=2)
