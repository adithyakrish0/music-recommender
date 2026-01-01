import os
import time
import json
import random
import logging
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from flask import Flask, render_template, request, jsonify, session
from flask_caching import Cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_harmonic_recommender')

# Configure cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Spotify Credentials
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '8a1446211a6648adb16de14a18991937')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '14a27936b1a548a79cef56500fcec1f3')

class SpotifyClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry = 0
    
    def get_token(self):
        if self.token and time.time() < self.token_expiry:
            return self.token
        
        try:
            auth_url = 'https://accounts.spotify.com/api/token'
            auth_response = requests.post(
                auth_url,
                auth=HTTPBasicAuth(self.client_id, self.client_secret),
                data={'grant_type': 'client_credentials'},
                timeout=5
            )
            auth_response.raise_for_status()
            data = auth_response.json()
            self.token = data.get('access_token')
            self.token_expiry = time.time() + data.get('expires_in', 3600) - 60
            return self.token
        except Exception as e:
            logger.error(f"Failed to get Spotify token: {e}")
            return None

    def get_album_art(self, track_name, artist_name):
        token = self.get_token()
        if not token:
            return None
        
        try:
            headers = {'Authorization': f'Bearer {token}'}
            # Search specifically for track and artist to be more accurate
            q = f"track:{track_name} artist:{artist_name}"
            search_url = 'https://api.spotify.com/v1/search'
            params = {
                'q': q,
                'type': 'track',
                'limit': 1
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                items = data.get('tracks', {}).get('items', [])
                if items:
                    images = items[0].get('album', {}).get('images', [])
                    if images:
                        return images[0]['url']
        except Exception as e:
            logger.error(f"Error fetching album art for {track_name}: {e}")
        return None

class MusicRecommender:
    def __init__(self, data_path):
        self.df = None
        # Comprehensive audio features
        self.audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.vectorizer = None
        self.scaler = None
        self.tfidf_matrix = None
        self.audio_matrix = None
        self.feature_matrix = None
        self.nn_model = None
        self.load_data(data_path)
        self.prepare_features()

    def load_data(self, path):
        try:
            self.df = pd.read_csv(path)
            
            # Standardize column names
            rename_map = {
                'track_name': 'name',
                'artists': 'artist',
                'track_genre': 'genre'
            }
            self.df.rename(columns={k: v for k, v in rename_map.items() if k in self.df.columns}, inplace=True)

            # Clean data
            cols_to_check = ['name', 'artist', 'genre'] + [f for f in self.audio_features if f in self.df.columns]
            self.df.dropna(subset=cols_to_check, inplace=True)

            # Clean artist names
            self.df['artist'] = self.df['artist'].astype(str).str.split(';').str[0]
            self.df['artist'] = self.df['artist'].str.replace(r'[\[\]\'"]', '', regex=True)

            # Handle popularity
            if 'popularity' not in self.df.columns:
                self.df['popularity'] = 50
            else:
                self.df['popularity'] = self.df['popularity'].clip(0, 100)

            # Create text features for search
            self.df['text_features'] = (
                self.df['name'].astype(str) + ' ' +
                self.df['artist'].astype(str) + ' ' +
                self.df['genre'].astype(str)
            ).str.lower()

            # Reset index to ensure alignment
            self.df = self.df.reset_index(drop=True)
            self.df['id_internal'] = self.df.index  # Use internal index for lookups

            logger.info(f"Loaded {len(self.df)} songs")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create minimal fallback dataframe
            self.df = pd.DataFrame([{
                'name': f"Song {i}", 'artist': "Artist", 'genre': "Pop",
                'popularity': 50, 'text_features': f"song {i} artist pop",
                **{feat: 0.5 for feat in self.audio_features}
            } for i in range(10)])
            self.df['id_internal'] = self.df.index

    def prepare_features(self):
        try:
            # 1. Text Features (TF-IDF)
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text_features'])

            # 2. Audio Features (Scaling)
            # Ensure all audio features exist, fill missing with 0.5
            for feat in self.audio_features:
                if feat not in self.df.columns:
                    self.df[feat] = 0.5

            self.scaler = MinMaxScaler()
            self.audio_matrix = self.scaler.fit_transform(self.df[self.audio_features])

            # 3. Combined Features for Content-Based Filtering (Likes)
            # We weight audio features more for "vibe" similarity
            # Stack sparse tfidf and dense audio
            self.feature_matrix = hstack([
                self.audio_matrix * 0.8,         # 80% Audio
                self.tfidf_matrix * 0.2          # 20% Text (Genre/Artist similarity)
            ]).tocsr()

            # 4. Initialize Nearest Neighbors
            self.nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
            self.nn_model.fit(self.feature_matrix)

            logger.info("Feature engineering complete.")

        except Exception as e:
            logger.error(f"Error in feature preparation: {e}")

    def search(self, query, n=20):
        """
        Search for songs by text query.
        Returns songs sorted by relevance and popularity.
        """
        if not query:
            return []

        try:
            query_vec = self.vectorizer.transform([query.lower()])

            # specific to text search, we only use tfidf_matrix
            cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Get top matches
            # We filter for relevance > 0
            relevant_indices = np.where(cosine_sim > 0)[0]

            if len(relevant_indices) == 0:
                return []

            # Sort by score
            sorted_indices = relevant_indices[np.argsort(cosine_sim[relevant_indices])[::-1]]

            # Take top 50 candidates, then sort by popularity to bubble up hits
            candidates = sorted_indices[:50]

            # Create a score that combines text match (high weight) and popularity (low weight)
            # Normalize popularity to 0-1
            pop_scores = self.df.iloc[candidates]['popularity'].values / 100.0
            text_scores = cosine_sim[candidates]

            final_scores = text_scores * 0.8 + pop_scores * 0.2

            # Re-sort candidates by final score
            final_top_indices = candidates[np.argsort(final_scores)[::-1]][:n]

            return self.df.iloc[final_top_indices].to_dict('records')

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def recommend_from_likes(self, liked_indices, n=10):
        """
        Recommend songs based on a list of liked song INDICES (internal ids).
        """
        if not liked_indices:
            return self.get_popular(n)

        try:
            # Create a user profile vector (average of liked songs)
            user_profile = self.feature_matrix[liked_indices].mean(axis=0)

            # Reshape to (1, n_features)
            user_profile = np.asarray(user_profile).reshape(1, -1)

            # Find neighbors
            # We ask for more neighbors to filter out the ones already liked
            n_neighbors = n + len(liked_indices) + 5
            distances, indices = self.nn_model.kneighbors(user_profile, n_neighbors=n_neighbors)

            recommendations = []
            for idx in indices[0]:
                if idx not in liked_indices:
                    recommendations.append(self.df.iloc[idx].to_dict())
                    if len(recommendations) >= n:
                        break

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self.get_popular(n)

    def get_popular(self, n=10):
        return self.df.sort_values('popularity', ascending=False).head(n).to_dict('records')

    def get_random_popular(self, n=10, top_k=200):
        """Get random songs from the top K popular songs"""
        top_songs = self.df.sort_values('popularity', ascending=False).head(top_k)
        return top_songs.sample(min(n, len(top_songs))).to_dict('records')

# Initialize components
recommender = MusicRecommender("spotify.csv")
spotify_client = SpotifyClient(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
@cache.cached(timeout=60, query_string=True)
def recommend():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify([])
    
    results = recommender.search(query)

    # Enrich with images
    # We do this asynchronously or on-demand usually, but for now we do it here for top 5
    # to keep it snappy, maybe just first 3?
    for song in results[:5]:
        song['image_url'] = spotify_client.get_album_art(song['name'], song['artist'])

    return jsonify(results)

@app.route('/swiper')
def swiper_view():
    return render_template('swiper.html')

@app.route('/api/swiper')
def swiper_data():
    # Reset session for new swiper round
    session['liked_indices'] = []

    # Get a pool of songs to swipe on
    # Mix of popular and random
    songs = recommender.get_random_popular(n=10)

    # Pre-fetch images for swiper (crucial for UX)
    for song in songs:
        song['image_url'] = spotify_client.get_album_art(song['name'], song['artist'])
        # If no image, use placeholder
        if not song.get('image_url'):
             song['image_url'] = f"https://via.placeholder.com/300x300/121212/FFFFFF?text={song['name'][0]}"

    # Store the pool in session mapping id to internal index if needed
    # actually we just need to send them to frontend.
    # Frontend sends back "id" or some identifier.
    # We used "id_internal" in the dataframe.

    return jsonify(songs)

@app.route('/swipe', methods=['POST'])
def swipe():
    data = request.get_json()
    song_id = data.get('song_id') # This should be the index or some unique ID

    if song_id is not None and song_id != -1:
        # User liked this song
        # In our simplified model, we trust the frontend sends back the `id_internal` or we find it.
        # But `records` output of pandas might not include index unless we made it a column.
        # We did: self.df['id_internal'] = self.df.index
        
        try:
            idx = int(song_id)
            session.setdefault('liked_indices', []).append(idx)
        except:
            pass
            
    # If frontend asks for recommendations (sent song_id -1 or logic handling)
    # The frontend logic in original was: after 10 swipes, call endpoint.
    # Here we can just return recommendations if we have enough likes or if requested.

    # If this is a "finish" call (e.g. song_id == -1)
    if song_id == -1 or len(session.get('liked_indices', [])) >= 5: # Threshold
        recs = recommender.recommend_from_likes(session.get('liked_indices', []), n=10)

        # Fetch images for recommendations
        for song in recs:
             song['image_url'] = spotify_client.get_album_art(song['name'], song['artist'])

        return jsonify({"recommendations": recs})
    
    return jsonify({"status": "ok", "liked_count": len(session.get('liked_indices', []))})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
