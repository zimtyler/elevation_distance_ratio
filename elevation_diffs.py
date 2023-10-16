import random
import requests
import numpy as np
import pandas as pd
import json
import os
from datetime import date
from math import radians, cos, sin, asin, sqrt

def returnAuth(auth_key):
    with open("sample_org.json") as file:
        orgData = json.load(file)
        apiKey = ordData[auth_key]["api_key"]
    return apiKey
    
def generate_points(csv, num_loc, random_seed=12):
    area_dict = {}
    df = pd.read_csv(csv)
    tups = df.itertuples(index=False, name=None)
  
    for tup in tups:
        # set seed to return same random vals for each iteration below -- want to measure variability using consistent differences in lat-long
        random.seed(random_seed) 
      
        area_dict[tup[0]] = {}
        area_dict[tup[0]]["epicenterLatLongs"] = (tup[1], tup[2])
        area_dict[tup[0]]["nearbyLatLongs"] = []
      
        for i in range(num_loc):
            lat_change = random.uniform(-1, 1)/2
            long_change = random.uniform(-1, 1)/2
            area_dict[tup[0]]["nearbyLatLongs"].append((tup[1]+lat_change, tup[2]+long_change))
    return area_dict

def get_url(area_dict, auth_key):
    urlList = []
    for key in area_dict.keys():
        lat_long_values = str(area_dict[key]["epicenterLatLongs"][0]) + '%2C' + str(area_dict[key]["epicenterLatLongs"][1])
        
        for tup in area_dict[key]["nearbyLatLongs"]:
            lat_long_values += '%7C' + str(tup[0]) + '%2C' + str(tup[1])
    
        urlList.append([key, f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat_long_values}&key={auth_key}"])
    return urlList

def return_results(urlList):
    elevation_dict = {}
    for l in urlList:
        response = requests.get(l[1])
        data = response.json()
        results = data["results"]
        elevation_dict[l[0]] = results
    return elevation_dict

def haversine(lon1, lat1, lon2, lat2, r=6371):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    r is set to return km.
    Set r to 3956 for miles.
    """
    # degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    dist = c * r
    return dist

def distance_from_epicenter(data):
    geo_dict = {}

    for dealer, list_ in data.items():
        geo_dict[dealer] = []

        for i, geoData in enumerate(list_):
            if i == 0:
                dealerElevation = float(geoData['elevation'])
                dealerLat = geoData['location']['lat']
                dealerLong = geoData['location']['lng']
            else:
                if geoData['elevation'] <= -2: # Lowest reasonable point on land (Lousisiana). Excluding Death Valley
                    continue
                else:
                    nested_dict = {}
                    # return abs diff between dealer and generated point (in meters)
                    nested_dict['elevation_diff'] = abs(geoData['elevation'] - dealerElevation)
                    # return haversine distance (in meters)
                    nested_dict['haversine_dist'] = haversine(dealerLat, dealerLong, geoData['location']['lat'], geoData['location']['lng']) * 1000 # elevation was measured in meters. Keep measures consistent
                    # return abs val of elevation difference

                    geo_dict[dealer].append(nested_dict)
    return geo_dict

def elevationDiscripDictList(geo_dict):
    geoList = []
    for id, list_ in geo_dict.items():

        elevation_array = np.array([float(x['elevation_diff']) for x in list_ if x != np.NaN])
        dist_mean = np.array([float(x['haversine_dist']) for x in list_ if x != np.NaN]).mean()
        
        dict_ = {
            "id": id,
            "avg_dist": dist_mean,
            "num_points": len(elevation_array),
            "avg_diff_elevation": elevation_array.mean(),
            "var_diff_elevation": elevation_array.var(),
            "stdev_diff_elevation": elevation_array.std()
        }
      
        geoList.append(dict_)
    return geoList
    
def main():
    path_to_csv = input("Path to csv: ")
    num_locs = input("Provide number of lat-long pairs to generate. Limit 60: ")
    
    if num_locs > 60:
        num_locs = 60
    
    auth_key = input("Provide Unique Org Id: ")
    api_key = returnAuth(auth_key)
    area_dict = generate_points(path_to_csv, 30, random_seed=12)
    urls = getUrl(area_dict, api_key)
    elev_dict = return_results(urls)
    
    rel_elev_dict = distance_from_epicenter(elev_dict)
    DescriptStatsList = elevationDiscripDictList(relElv_dict)
    
    today = str(date.today()).replace("-", "_")
    format = input("Save to csv[1] or json[2]? ")
    targetFilePath = f"targetFilePath/elevation_diff_descripStats{today}"
    
    if format == 1:
        df = pd.DataFrame(DescriptStatsList)
        df.to_csv(targetFilePath + ".csv")
    else:
        targetFilePath += ".json"
        with open(targetFilePath, "w") as file:
            json.dump(DescriptStatsList, file, indent=2)


if __name__ == "__main__":
    main()
        
        
    
    
  
