import unittest
import numpy as np
from app import MusicRecommender

class TestMusicRecommender(unittest.TestCase):
    def setUp(self):
        # Initialize with the real file but we'll check basic functionality
        self.recommender = MusicRecommender("spotify.csv")

    def test_search_results(self):
        # Search for something known
        results = self.recommender.search("pop", n=5)
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("pop" in r['genre'].lower() for r in results))

    def test_recommend_from_likes(self):
        # Pick some random indices
        liked_indices = [0, 1, 2]
        recs = self.recommender.recommend_from_likes(liked_indices, n=5)
        self.assertEqual(len(recs), 5)
        # Ensure no duplicates from liked
        rec_names = [r['name'] for r in recs]
        liked_names = [self.recommender.df.iloc[i]['name'] for i in liked_indices]
        for name in liked_names:
            self.assertNotIn(name, rec_names)

if __name__ == '__main__':
    unittest.main()
