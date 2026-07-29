#%%
import requests
import pandas as pd

def get_tle_json(base_url, norad_id=25544):
    #norad_id = 25544
    #base_url = "https://satchecker.cps.iau.org"
    url = f"{base_url}/tools/tles-at-epoch/"
    # 1. Get catalog
    params = {
        "id": norad_id,
        "id_type": "catalog",
        "sort": "-epoch",
        "page_size": 1
    }
    
    data = requests.get(url).json()["data"]

    # 2. Show options
    for i, sat in enumerate(data[:20]):
        print(i, sat["satellite_name"], sat["satellite_id"])

    # 3. Select one
    choice = int(input("Select index: "))
    selected = data[choice]

    name = selected["satellite_name"]
    norad_id = selected["satellite_id"]

    print("Selected:", name, norad_id)

    # 4. Get TLE (your code)
    tle_url = "https://satchecker.cps.iau.org/tools/get-tle-data/"
    

    tle_data = requests.get(tle_url, params=params).json()["data"][0]

    print(tle_data["tle_line1"])
    print(tle_data["tle_line2"])

def get_tles_at_epoch(base_url, epoch_date=None, page=1, per_page=100):
    url = f"{base_url}/tools/tles-at-epoch/"
    params = {
        "epoch": epoch_date,
        "page": page,
        "per_page": per_page
    }
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def fetch_all_tles(base_url, epoch_date=None, data_source="spacetrack"):
    all_tles = []
    page = 1
    per_page = 100

    while True:
        results = get_tles_at_epoch(base_url, epoch_date, page, per_page)
        tles = results[0]["data"]
        all_tles.extend(tles)
        if len(tles) < per_page:
            break
        page += 1

    return all_tles


if __name__ == "__main__":
    base_url = "https://satchecker.cps.iau.org"

    # This will give the current TLE set, use a specific epoch (in Julian date format) if needed
    epoch_date = 2460606
    get_tle_json(base_url, norad_id=25544)
   


    # all_tles = fetch_all_tles(base_url, epoch_date=epoch_date)

    # # Create a DataFrame from the TLE data
    # df = pd.DataFrame(all_tles)
    # print(df.columns)
    # print(df.head())
    # print(df.shape[0])
# print(data)
#%%

# import requests
# import csv

# USERNAME = "bayajid.khan@upm.es"
# PASSWORD = "Space_track_Gedanken1"


# login_url = "https://www.space-track.org/ajaxauth/login"

# query_url = (
#     "https://www.space-track.org/basicspacedata/query/"
#     "class/satcat/CURRENT/Y/DECAY/null-val/"
#     "orderby/NORAD_CAT_ID/format/json"
# )

# session = requests.Session()

# # Login
# session.post(
#     login_url,
#     data={
#         "identity": USERNAME,
#         "password": PASSWORD
#     },
#     timeout=30
# )

# # Fetch satellite catalog
# response = session.get(query_url, timeout=60)
# response.raise_for_status()

# data = response.json()

# # Keep only NORAD ID + satellite name
# rows = []

# for item in data:
#     rows.append({
#         "NORAD_CAT_ID": item["NORAD_CAT_ID"],
#         "OBJECT_NAME": item["OBJECT_NAME"]
#     })

# df = pd.DataFrame(rows)

# # Save CSV
# df.to_csv("active_satellites.csv", index=False)

# print(df.head())
# print(f"\nSaved {len(df)} satellites")
#%%
from MEKF import MEKFComparator, QuaternionMath

#MEKFComparator import 
import numpy as np


def quaternion_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular error between two quaternions (radians), taking the shortest path."""
    q1 = QuaternionMath.normalize(q1)
    q2 = QuaternionMath.normalize(q2)
    if np.dot(q1, q2) < 0:
        q2 = -q2
        print("Quaternions are on opposite hemispheres, negating one to ensure shortest path.")
    else:
        print("Quaternions are on the same hemisphere, no negation needed.")
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)

q_true=   np.array([0.7071068, 0.0000000, 0.0000000, -0.7071068])
q_est =   np.array([0.7075299, -0.0092841, -0.0029685, -0.7066162])

res = np.array(quaternion_error(q_true, q_est))

print(res)
# %%
