from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import folium

# Load the KMeans model
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["data"]
    data = np.array(data).reshape(1, -1)
    cluster = kmeans_model.predict(data)
    return jsonify({"cluster": int(cluster[0])})

@app.route('/map')
def map_view():
    # Example coordinates for the center of the map (can be modified as needed)
    map_center = [-1.286389, 36.817223]  # Nairobi
    folium_map = folium.Map(location=map_center, zoom_start=7)

    # Plot clusters on the map
    locations = [
        # Add real data points with latitude and longitude and cluster assignments
        {"lat": -1.2921, "lon": 36.8219, "cluster": 0},  # Example point for Cluster 0
        {"lat": -1.286389, "lon": 36.817223, "cluster": 1},  # Example point for Cluster 1
    ]

    # Use different colors for each cluster
    colors = {0: "yellow", 1: "blue"}
    for loc in locations:
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"Cluster {loc['cluster']}",
            icon=folium.Icon(color=colors[loc["cluster"]])
        ).add_to(folium_map)

    # Save the map as an HTML file
    folium_map.save("templates/map.html")
    return render_template("map.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
