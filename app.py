from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import csv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key')

# Spotify configuration - Updated to handle both environments
SPOTIPY_CLIENT_ID = os.environ['SPOTIPY_CLIENT_ID']
SPOTIPY_CLIENT_SECRET = os.environ['SPOTIPY_CLIENT_SECRET']
SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI', 'http://localhost:5000/callback')
SCOPE = "user-library-read user-top-read"

# --- Load the Spotify CSV Data ---
try:
    df = pd.read_csv("spotify.csv")
    df.rename(columns={
        'track_name': 'name', 
        'artists': 'artist', 
        'track_genre': 'genre'
    }, inplace=True)
except Exception as e:
    print(f"❌ Error loading Spotify CSV: {e}")
    df = pd.DataFrame()  # Fallback empty DataFrame

def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE,
        show_dialog=True  # Added to make auth flow clearer
    )

# --- Home/Login Route ---
@app.route('/')
def home():
    if 'spotify_token' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

# --- Handle Form Login ---
@app.route('/login', methods=['POST'])
def login():
    # Save user info to users.csv
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']

    user_file = 'users.csv'
    file_exists = os.path.isfile(user_file)

    with open(user_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Email', 'Phone'])  # Header
        writer.writerow([name, email, phone])

    return redirect('/main')

# --- Spotify OAuth Login ---
@app.route('/spotify_login')
def spotify_login():
    try:
        sp_oauth = create_spotify_oauth()
        auth_url = sp_oauth.get_authorize_url()
        print(f"Redirecting to Spotify auth URL: {auth_url}")  # Debug log
        return redirect(auth_url)
    except Exception as e:
        print(f"Error in spotify_login: {e}")
        return redirect(url_for('home'))

# --- Spotify OAuth Callback ---
@app.route('/callback')
def callback():
    try:
        sp_oauth = create_spotify_oauth()
        print(f"Callback received. Redirect URI: {SPOTIPY_REDIRECT_URI}")  # Debug log
        token_info = sp_oauth.get_access_token(request.args['code'])
        session['spotify_token'] = token_info
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error in callback: {e}")
        return redirect(url_for('home'))

# --- Dashboard for Spotify Users ---
@app.route('/dashboard')
def dashboard():
    if 'spotify_token' not in session:
        return redirect(url_for('home'))
    
    try:
        sp = spotipy.Spotify(auth=session['spotify_token']['access_token'])
        user = sp.current_user()
        top_tracks = sp.current_user_top_tracks(limit=10)
        return render_template('dashboard.html', user=user, top_tracks=top_tracks)
    except Exception as e:
        print(f"Spotify API error: {e}")
        session.pop('spotify_token', None)
        return redirect(url_for('home'))

# --- Main Page (Recommendation UI) ---
@app.route('/main')
def main():
    return render_template('index.html')

# --- Song Search API ---
@app.route('/search', methods=['GET'])
def search_songs():
    artist = request.args.get('artist', '').strip().lower()
    genre = request.args.get('genre', '').strip().lower()

    filtered_df = df.copy()

    if artist:
        filtered_df = filtered_df[filtered_df['artist'].str.lower().str.contains(artist, na=False)]
    if genre:
        filtered_df = filtered_df[filtered_df['genre'].str.lower().str.contains(genre, na=False)]

    if filtered_df.empty:
        return {'message': 'No matching songs found.'}

    if 'popularity' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by="popularity", ascending=False)

    results = filtered_df[['name', 'artist', 'genre', 'popularity']].to_dict(orient="records")
    return results

# --- Logout ---
@app.route('/logout')
def logout():
    session.pop('spotify_token', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Print debug info about Spotify config
    print(f"Spotify Client ID: {SPOTIPY_CLIENT_ID}")
    print(f"Spotify Redirect URI: {SPOTIPY_REDIRECT_URI}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)