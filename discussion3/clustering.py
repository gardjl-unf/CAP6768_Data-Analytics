# Author: Jason Gardner
# Date: 09/22/2024
# Class: CAP6768
# Assignment: Discussion 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as ac
from sklearn.cluster import KMeans as km
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.ticker as ticker
import geopandas as gpd
from shapely.geometry import Polygon


STATES_SHAPEFILE_PATH = "./map_data/ne_110m_admin_1_states_provinces_lakes.shp"
US_SHAPEFILE_PATH = "./map_data/ne_110m_admin_0_countries.shp"
OUTPUT_FILE = "Analysis.txt"
us_states = gpd.read_file(STATES_SHAPEFILE_PATH)
world = gpd.read_file(US_SHAPEFILE_PATH)
us_country = world[world['SOVEREIGNT'] == 'United States of America']
alaska_box = Polygon([(-179, 51), (-179, 71), (-129, 71), (-129, 51)])
puerto_rico_box = Polygon([(-70, 17), (-70, 19), (-65, 19), (-65, 17)])
alaska_mask = gpd.GeoDataFrame(geometry=[alaska_box], crs=us_country.crs)
puerto_rico_mask = gpd.GeoDataFrame(geometry=[puerto_rico_box], crs=us_country.crs)
us_country = us_country.overlay(alaska_mask, how='difference')
us_country = us_country.overlay(puerto_rico_mask, how='difference')

FILENAME = "NCAA Schools.csv"
CATEGORY = "School"
CLUSTERS = 10
GROUP_NAMES = ["Group 1", 
               "Group 2", 
               "Group 3", 
               "Group 4", 
               "Group 5", 
               "Group 6", 
               "Group 7", 
               "Group 8", 
               "Group 9", 
               "Group 10"]
GRAPH_TITLES = ["Group Average Linkage Clusters", 
                "Complete Linkage Clusters", 
                "Ward's Method Linkage Clusters", 
                "Centroid Linkage Clusters"]
COLORS = plt.cm.rainbow(np.linspace(0, 1, CLUSTERS))

class Data:
    def __init__(self) -> None:
        self.filename = FILENAME
        self.data = self.load_data()
        self.data = self._process_data()
        
    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.filename)
        return data

    def _process_data(self) -> np.ndarray:
        X = self.data["Longitude"].values
        Y = self.data["Latitude"].values
        data = np.column_stack((X, Y))
        return data
    
    def get_data(self) -> np.ndarray:
        return self.data
    
def analyze_clusters(model, data, method_name):
    labels = model.labels_
    unique_labels = np.unique(labels)
    
    cluster_info = []
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_size = len(cluster_points)
        min_lat, min_lon = np.min(cluster_points, axis=0)
        max_lat, max_lon = np.max(cluster_points, axis=0)
        centroid_lat = np.mean(cluster_points[:, 1])
        centroid_lon = np.mean(cluster_points[:, 0])
        std_lat = np.std(cluster_points[:, 1])
        std_lon = np.std(cluster_points[:, 0])
        box_size = (max_lat - min_lat) * (max_lon - min_lon)
        
        if cluster_size > 1:
            avg_distance = np.mean([np.linalg.norm(cluster_points[i] - cluster_points[j]) for i in range(cluster_size) for j in range(i + 1, cluster_size)])
        else:
            avg_distance = 0
        
        cluster_info.append({
            "Cluster Method": method_name,
            "Cluster": label + 1,
            "Size": cluster_size,
            "Min Lon": min_lon,
            "Max Lon": max_lon,
            "Min Lat": min_lat,
            "Max Lat": max_lat,
            "Centroid Lat": centroid_lat,
            "Centroid Lon": centroid_lon,
            "Std Lat": std_lat,
            "Std Lon": std_lon,
            "Box Size": box_size,
            "Avg Distance": avg_distance,
            "Isolated": "Yes" if cluster_size == 1 else "No"
        })
    
    return cluster_info



def plot_dendrogram(data, cluster_method, method_name):
    linked = linkage(data, method=cluster_method)
    
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title(f"Dendrogram for {method_name}")
    plt.xlabel("Schools")
    plt.ylabel("Euclidean Distance")
    plt.savefig(f"{method_name} Dendrogram.png", bbox_inches='tight')

if __name__ == "__main__":
    data = Data().get_data()
    models = []
    all_cluster_data = []

    for cluster_method, graph_title in zip(["average", "complete", "ward", "centroid"], GRAPH_TITLES):
        if cluster_method == "centroid":
            model = km(n_clusters=CLUSTERS)
        else:
            model = ac(n_clusters=CLUSTERS, metric='euclidean', linkage=cluster_method)
        
        models.append(model)
        model.fit(data)

        cluster_data = analyze_clusters(model, data, cluster_method)
        all_cluster_data.extend(cluster_data)

        fig, ax = plt.subplots(figsize=(12, 5))
        us_states.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.8, zorder=2, linewidth=0.5)
        us_country.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.8, zorder=1, linewidth=0.75)

        for label in np.unique(model.labels_):
            plt.scatter(data[model.labels_ == label, 0], data[model.labels_ == label, 1], 
                        label=f"Group {label+1}", color=COLORS[label], zorder=3)

        ax.set_facecolor("#f0f0f0")
        ax.set_xlabel("Longitude", fontsize=12, fontweight='bold')
        ax.set_ylabel("Latitude", fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlim(-165, -65)
        ax.set_ylim(15, 50)
        ax.grid(True, which='both')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax.grid(which='major', linestyle='-', linewidth='1')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[i], markersize=10) for i in range(CLUSTERS)]
        legend = ax.legend(handles, GROUP_NAMES, title="Groups", title_fontsize=14, fontsize=10, shadow=True, 
                           facecolor="white", edgecolor="black", loc="upper left")
        legend.get_title().set_fontweight('bold')
        ax.set_title(graph_title, fontsize=14, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.savefig(f"{graph_title}.png", bbox_inches='tight')
        plt.close()
        
    for cluster_method, graph_title in zip(["average", "complete", "ward"], GRAPH_TITLES):  
        plot_dendrogram(data, cluster_method, graph_title)
    
    cluster_df = pd.DataFrame(all_cluster_data)
    cluster_df.to_csv("cluster_analysis.csv", index=False)
    cluster_df_sorted = cluster_df.sort_values(by=["Cluster Method", "Min Lon"])
    cluster_df_sorted.to_csv("cluster_analysis_sorted.csv", index=False)
