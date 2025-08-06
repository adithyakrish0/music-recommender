from flask import Flask, render_template, request, jsonify, session
from flask_caching import Cache
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import requests
from requests.auth import HTTPBasicAuth
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree, NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a real secret key

# Configure cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Spotify API credentials
SPOTIFY_CLIENT_ID = '8a1446211a6648adb16de14a18991937'
SPOTIFY_CLIENT_SECRET = '14a27936b1a548a79cef56500fcec1f3'

# Load music data with better error handling
try:
    df = pd.read_csv("spotify.csv")
    
    # Standardize column names
    df.rename(columns={
        'track_name': 'name', 
        'artists': 'artist', 
        'track_genre': 'genre'
    }, inplace=True)
    
    # Clean data
    df.dropna(subset=['name', 'artist', 'genre'], inplace=True)
    df['artist'] = df['artist'].str.split(';').str[0]
    df['artist'] = df['artist'].str.replace(r'[\[\]\'"]', '', regex=True)
    
    # Handle popularity
    if 'popularity' not in df.columns:
        df['popularity'] = 50
    else:
        df['popularity'] = df['popularity'].clip(0, 100)
    
    # Create features for recommendation
    df['combined_features'] = (df['artist'] + ' ' + df['genre'] + ' ' + 
                             df['name'] + ' ' + 
                             df['danceability'].astype(str) + ' ' +
                             df['energy'].astype(str))
    
    print(f"Loaded {len(df)} songs") 
except Exception as e:
    print(f"Error loading data: {e}")
    # Create fallback dataframe
    df = pd.DataFrame([{
        'id': i+1,
        'name': f"Sample Song {i+1}", 
        'artist': "Sample Artist", 
        'genre': "Pop",
        'popularity': 50
    } for i in range(10)])

tfidf_matrix = None
similarity_matrix = None
feature_matrix = None
vectorizer = None

def initialize_recommendation_engine():
    global feature_matrix, vectorizer, df  # Add 'df' to the global declaration
    try:
        # Only vectorize text features
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
        
        # Select and scale audio features
        audio_features = ['danceability', 'energy', 'valence', 'tempo']
        audio_scaler = MinMaxScaler()
        audio_matrix = audio_scaler.fit_transform(df[audio_features])
        
        # Combine features (weight audio more heavily)
        feature_matrix = csr_matrix(np.hstack([
            audio_matrix * 0.7,  # 70% weight to audio features
            tfidf_matrix.toarray() * 0.3  # 30% weight to text
        ]))
        
        print("Recommendation engine initialized successfully with sparse matrix")
    except Exception as e:
        print(f"Error initializing recommendation engine: {e}")
        # Fallback to small random sample if initialization fails
        df = df.sample(1000).reset_index(drop=True)
        initialize_recommendation_engine()

def prepare_features(df):
    """Prepare audio features for recommendation engine"""
    # Select relevant audio features
    audio_features = ['danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 
                    'instrumentalness', 'liveness', 
                    'valence', 'tempo']
    
    # Normalize features
    scaler = MinMaxScaler()
    df[audio_features] = scaler.fit_transform(df[audio_features])
    
    # Combine with text features
    tfidf = vectorizer.fit_transform(df['combined_features'])
    audio_matrix = df[audio_features].values
    
    # Weight audio features more heavily than text
    return np.hstack([audio_matrix * 0.7, tfidf.toarray() * 0.3])

def get_recommendations_based_on_likes(liked_song_ids, n=10):
    try:
        if not liked_song_ids or len(liked_song_ids) == 0:
            return get_popular_fallback(n)
            
        # Get indices of liked songs
        liked_indices = df[df['id'].isin(liked_song_ids)].index
        
        # Use BallTree for efficient nearest neighbor search
        tree = BallTree(feature_matrix, metric='euclidean')
        
        # Get average features of liked songs
        avg_features = feature_matrix[liked_indices].mean(axis=0)
        
        # Find similar songs (query 2x more than needed to filter out liked songs)
        _, indices = tree.query(avg_features, k=min(n*2, len(df)))
        
        # Filter recommendations
        recommendations = []
        for idx in indices[0]:
            song = df.iloc[idx].to_dict()
            if song['id'] not in liked_song_ids and len(recommendations) < n:
                recommendations.append(song)
        
        return recommendations if recommendations else get_popular_fallback(n)
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        return get_popular_fallback(n)

def get_popular_fallback(n):
    """Fallback to popular songs when recommendations fail"""
    return (df.sort_values('popularity', ascending=False)
            .head(n)
            .to_dict('records'))

def get_hybrid_recommendations(query, n=10):
    """Combine search and audio features for better results"""
    try:
        # Text-based search first
        query_vec = vectorizer.transform([query])
        text_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Audio feature similarity
        features = prepare_features(df)
        knn = NearestNeighbors(n_neighbors=n*2, metric='cosine')
        knn.fit(features)
        _, audio_indices = knn.kneighbors(features.mean(axis=0).reshape(1, -1))
        
        # Combine results
        combined_scores = []
        for idx in range(len(df)):
            score = text_scores[idx] * 0.4  # Weight text less
            if idx in audio_indices:
                audio_rank = np.where(audio_indices[0] == idx)[0]
                if len(audio_rank) > 0:
                    score += (1 - audio_rank[0]/len(audio_indices[0])) * 0.6
            combined_scores.append(score)
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:n]
        return df.iloc[top_indices].to_dict('records')
    except Exception as e:
        print(f"Hybrid recommendation error: {e}")
        return get_popular_fallback(n)

def get_spotify_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(
        auth_url,
        auth=HTTPBasicAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        data={'grant_type': 'client_credentials'}
    )
    return auth_response.json().get('access_token')

def get_album_art(song_name, artist_name):
    try:
        token = get_spotify_token()
        headers = {'Authorization': f'Bearer {token}'}
        search_url = f'https://api.spotify.com/v1/search?q=track:{song_name} artist:{artist_name}&type=track&limit=1'
        response = requests.get(search_url, headers=headers).json()
        
        if response.get('tracks', {}).get('items'):
            return response['tracks']['items'][0]['album']['images'][0]['url']  # Largest image
    except Exception as e:
        print(f"Error fetching album art: {e}")
    return None

# --- Main Page ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Recommendation API ---
@app.route('/recommend')
@cache.cached(timeout=300, query_string=True)
def recommend():
    query = request.args.get('query', '').lower().strip()
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        results = get_hybrid_recommendations(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Swiper Interface ---
@app.route('/swiper')
def swiper_view():
    """Render the swiper interface template"""
    return render_template('swiper.html')

@app.route('/api/swiper')
def swiper_data():
    try:
        if df.empty:
            raise ValueError("No data available")
            
        if 'popularity' not in df.columns:
            df['popularity'] = 50
            
        popular = df.sort_values('popularity', ascending=False).head(100)
        swipe_pool = popular.sample(min(10, len(popular))).to_dict('records')
        
        # Add image URLs to each song
        token = get_spotify_token()
        for song in swipe_pool:
            song['image_url'] = get_album_art(song['name'], song['artist'])
        
        session['swipe_pool'] = swipe_pool
        session['liked_songs'] = []
        session['disliked_songs'] = []
        
        return jsonify(swipe_pool)
    except Exception as e:
        print(f"Swiper error: {e}")
        sample_songs = [{
            "id": i+1,
            "name": f"Sample Song {i+1}",
            "artist": "Sample Artist",
            "genre": "Pop",
            "popularity": 50,
            "image_url": "https://via.placeholder.com/300"  # Fallback image
        } for i in range(10)]
        
        session['swipe_pool'] = sample_songs
        session['liked_songs'] = []
        session['disliked_songs'] = []
        
        return jsonify(sample_songs)

@app.route('/swipe', methods=['POST'])
def swipe():
    try:
        if 'swipe_pool' not in session:
            raise ValueError("Session expired - please refresh")
            
        data = request.get_json()
        if not data or 'song_id' not in data:
            raise ValueError("Invalid swipe data")
            
        # Track liked songs
        session.setdefault('liked_songs', []).append(int(data['song_id']))
        print(f"Liked songs: {session['liked_songs']}")
        
        # Check if we've completed all swipes
        if len(session.get('liked_songs', [])) + len(session.get('disliked_songs', [])) >= 10:
            recommendations = get_recommendations_based_on_likes(session['liked_songs'])
            
            # Clear session data
            session.pop('swipe_pool', None)
            session.pop('liked_songs', None)
            session.pop('disliked_songs', None)
            
            return jsonify({
                "recommendations": recommendations,
                "message": "Based on your preferences"
            })
            
        return jsonify({"status": "keep swiping"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/test-spotify')
def test_spotify():
    token = get_spotify_token()
    if not token:
        return "Failed to get Spotify token"
    
    # Test with a known song
    test_url = get_album_art("Blinding Lights", "The Weeknd")
    return jsonify({
        "token": token[:20] + "...",  # Don't expose full token
        "test_image_url": test_url
    })

if __name__ == '__main__':
    initialize_recommendation_engine()  # Initialize the recommendation engine
    app.run(debug=True, port=5000)