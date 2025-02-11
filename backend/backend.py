import os
import requests
import json
import uuid
import time
import numpy as np
from datetime import datetime, timedelta, timezone


'''
1. collect data of global sounding balloons and put them in their own buckets
2. I am planning to put em in seperate files too. This way we can later look at the data if we want to. We can also ignore previous trajectory history if we want to. 
3. Assuming that the average flight time of these balloons are ~400hrs we have to introduce a logic to dissociate trajectories start new ones. (Currently planning to do it with distance formula where if we start getting for a balloon in a bucket but the coordinate deviation is large enough for a long time, we can assume that it is a new trajectory all together). We can prolly use hashs here to give new names to every new balloon.
4. If well documented, my code with minimal changes should also allow people to manipulate trajectory buckets so that they can decide if they want to record them as part of the same flight history. (I am finding it difficult to think why one would want it, but imma keep a look at it anyway)
'''

CONSTELLATION_ENDPOINT = 'https://a.windbornesystems.com/treasure/'
DATA_DIR = "balloon_data"
UUID_FILE = os.path.join(DATA_DIR, "balloons.json")
DISTANCE_THRESHOLD_KM = 80  # based on this old https://www.weather.gov/media/key/KEY%20-%20Weather%20Balloon%20Poster.pdf data and assuming that I only loose contact with a balloon for say ~3hrs
# will change this to be dyanmic on history later

balloon_uuids = {}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if os.path.exists(UUID_FILE):
    with open(UUID_FILE, "r") as f:
        balloon_uuids = json.load(f)

def haversine(lat1, lon1, lat2, lon2):
    #gpt'd this part
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_history():
    for i in range(24):
        response = requests.get(f"{CONSTELLATION_ENDPOINT}{str(i).zfill(2)}.json")
        print(i, response)
        if response.status_code == 200:
            try:
                # for now i have only encountered one type of corruption where the json file doesnt begin with a '['
                # i will address more corruption possibilites later -- after i make decent progress overall
                if response.content.decode('utf8').splitlines()[0] != '[':
                    data = '[\n'+response.content.decode('utf8')
                else: data = response.content.decode('utf8')
                data = json.loads(data)
                timestamp = (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
                for balloon in data:
                    lat, lon, alt = balloon
                    found = False
                    for balloon_id, traj in balloon_uuids.items():
                        print(balloon_id, traj)
                        last_pos = traj[-1] if traj else None
                        if last_pos:
                            # gpt'd something called haversine distance
                            dist = haversine(last_pos[0], last_pos[1], lat, lon)
                            if dist < DISTANCE_THRESHOLD_KM:
                                balloon_uuids[balloon_id].append([lat, lon, alt, timestamp])
                                found = True
                                break

                    if not found:
                        new_id = str(uuid.uuid4())
                        balloon_uuids[new_id] = [[lat, lon, alt, timestamp]]
                            
            except Exception as e:
                #print(json.loads(response.content))
                print(f"Error while parsing: {e}")

            with open(UUID_FILE, "w") as f:
                json.dump(balloon_uuids, f)

            for balloon_id, trajectory in balloon_uuids.items():
                with open(os.path.join(DATA_DIR, f"{balloon_id}.json"), "w") as f:
                    json.dump(trajectory, f)


build_history()

