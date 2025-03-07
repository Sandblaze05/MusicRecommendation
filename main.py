import pandas as pd
from sklearn.neighbors import NearestNeighbors
import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

firebase_credentials = {
  "type": os.getenv("FIREBASE_TYPE"),
  "project_id": os.getenv("FIREBASE_PROJECT_ID"),
  "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
  "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
  "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
  "client_id": os.getenv("FIREBASE_CLIENT_ID"),
  "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
  "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
  "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
  "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
  "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
}


def get_similar_songs(track_ids, n=10, artist_bias=0.1):
    # Get indices of the given track IDs
    track_indices = tracks[tracks["track_id"].isin(track_ids)].index.tolist()

    if not track_indices:
        return pd.DataFrame(columns=["track_id", "track_name"])  # Return empty DataFrame if no matches

    # Compute average feature vector of selected tracks
    track_features = features.iloc[track_indices].mean(axis=0).values.reshape(1, -1)

    # Convert to DataFrame (preserving column names)
    track_features_df = pd.DataFrame(track_features, columns=features.columns)

    # Get similar songs
    distances, indices = nn_model.kneighbors(track_features_df, n_neighbors=n+1)

    similar_songs = tracks.iloc[indices[0][1:]].copy()
    original_artists = tracks.iloc[track_indices]["artists"].tolist()

    # Adjust distances based on artist match
    similar_songs["adjusted_distance"] = [
        d - artist_bias if a in original_artists else d
        for d, a in zip(distances[0][1:], similar_songs["artists"])
    ]

    # Sort by adjusted distance
    similar_songs = similar_songs.sort_values(by="adjusted_distance")

    return similar_songs.head(n)[["track_id", "track_name"]]

if __name__ == "__main__":

    # Initialize Firebase
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    user = 'gNOmy15xVWcVSiygtfdWfsG02682'

    favorites_ref = db.collection('users').document(user).collection('favorites').get()
    l=[]
    for fav in favorites_ref:
        l.append(fav.to_dict()['id']) if fav.to_dict()['type'] == 'Track' else None
    
    print(len(l))

    # Get the model ready
    tracks = pd.read_csv('tracks_cleaned.csv')
    
    features = tracks.drop(columns=["track_id", "track_name", "time_signature"])
    
    nn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
    nn_model.fit(features)
    
    similar_songs = get_similar_songs(l)
    print(similar_songs)
