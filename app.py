from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import requests
from requests.auth import HTTPBasicAuth

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a real secret key

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

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = None
similarity_matrix = None

def initialize_recommendation_engine():
    global tfidf_matrix, similarity_matrix
    try:
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        print("Recommendation engine initialized successfully")
    except Exception as e:
        print(f"Error initializing recommendation engine: {e}")

def get_recommendations_based_on_likes(liked_song_ids, n=5):
    try:
        liked_indices = df[df['id'].isin(liked_song_ids)].index
        if len(liked_indices) == 0:
            return df.sample(min(n, len(df))).to_dict('records')
        
        # Get average similarity scores
        avg_similarity = similarity_matrix[liked_indices].mean(axis=0)
        
        # Get top similar songs not already liked
        similar_indices = avg_similarity.argsort()[::-1]
        recommended_indices = [i for i in similar_indices if i not in liked_indices][:n]
        
        return df.iloc[recommended_indices].to_dict('records')
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return df.sample(min(n, len(df))).to_dict('records')

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
@app.route('/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query', '').lower().strip()
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        # First try exact matches
        exact_matches = df[
            (df['artist'].str.lower().str.contains(query, regex=False)) |
            (df['genre'].str.lower().str.contains(query, regex=False)) |
            (df['name'].str.lower().str.contains(query, regex=False))
        ]
        
        # If not enough exact matches, use TF-IDF similarity
        if len(exact_matches) < 5:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            similar_indices = similarities.argsort()[::-1][:10]
            similar_songs = df.iloc[similar_indices]
            results = pd.concat([exact_matches, similar_songs]).drop_duplicates()
        else:
            results = exact_matches
        
        return jsonify(results.sort_values('popularity', ascending=False)
                      .head(10)
                      .to_dict('records'))
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
            # Get recommendations based on liked songs
            liked_songs = session.get('liked_songs', [])
            
            # If no songs were liked, return popular songs
            if not liked_songs:
                recommended = df.sort_values('popularity', ascending=False).head(5).to_dict('records')
            else:
                # Get recommendations based on liked songs
                recommended = get_recommendations_based_on_likes(liked_songs)
            
            # Clear session data
            session.pop('swipe_pool', None)
            session.pop('liked_songs', None)
            session.pop('disliked_songs', None)
            
            return jsonify({
                "recommendations": recommended,
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