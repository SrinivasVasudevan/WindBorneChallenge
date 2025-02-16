import sqlite3
import sys
import requests
import json
import time
import uuid
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, send_file
from flask_cors import CORS

import threading
import matplotlib.pyplot as plt

import openmeteo_requests

import requests_cache
from retry_requests import retry

############################
'''
OPEN SOURCE API PARAMS
'''

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
####################



CONSTELLATION_ENDPOINT = "https://a.windbornesystems.com/treasure/"
DATA_DIR = "balloon_data"
DB_FILE = 'balloon_trajectories.db'
DISTANCE_THRESHOLD_KM = 50  # based on this old https://www.weather.gov/media/key/KEY%20-%20Weather%20Balloon%20Poster.pdf data and https://sites.wff.nasa.gov/code820/pages/about/about-faq.html#:~:text=The%20balloon%20typically%20rises%20at,120%2C000%20ft%20(36.6%20km).
# will change this to be dyanmic on history later
QUADRANT_SIZE = 10  # Define quadrant size for spatial partitioning

app = Flask(__name__)

'''
1. collect data of global sounding balloons and put them in their own buckets
2. I am planning to put em in seperate files too. This way we can later look at the data if we want to. We can also ignore previous trajectory history if we want to. 
3. Assuming that the average flight time of these balloons are ~400hrs we have to introduce a logic to dissociate trajectories start new ones. (Currently planning to do it with distance formula where if we start getting for a balloon in a bucket but the coordinate deviation is large enough for a long time, we can assume that it is a new trajectory all together). We can prolly use hashs here to give new names to every new balloon.
4. If well documented, my code with minimal changes should also allow people to manipulate trajectory buckets so that they can decide if they want to record them as part of the same flight history. (I am finding it difficult to think why one would want it, but imma keep a look at it anyway)
'''

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

'''
Some ideas i am trying out here: documenting so that i dont lose track of it
I want the trajectory builder to be robust enough to
a) register the fact that row 0 in 00.json need not always mean it is the latest trajectory for row 0 in 01.json (we can entertain that but maintain checks to ensure we have a robust trajectory in place)
b) be robust enough to handle entirely new balloon trajectory and mark old ones after some time as inactive -- active flag exactly for that reason
c) have a an optimization in place to facilitate a can happen by seggregating coordinates into quadrants and starting the search from there. we expand search only in case of no hits
'''
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balloon_id TEXT,
            rowno INTEGER,
            ass_rowno INTEGER,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            timestamp TIMESTAMP,
            active INTEGER DEFAULT 1,
            predicted INTEGER DEFAULT 0,
            corrupted INTEGER DEFAULT 0,
            possible_kin TEXT,
            strike INTEGRER DEFAULT 0,
            d_moved REAL
        )
    ''')
    conn.commit()
    conn.close()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = np.radians(lat2 - lat1)  
    dlon = np.radians(lon2 - lon1)  
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_quadrant(lat, lon):
    return int(lat // QUADRANT_SIZE), int(lon // QUADRANT_SIZE)

def linear_extrapolate(past_points, time_diff):
    if len(past_points) < 2:
        return None
    lat_diff = past_points[-1][0] - past_points[-2][0]
    lon_diff = past_points[-1][1] - past_points[-2][1]
    alt_diff = past_points[-1][2] - past_points[-2][2]
    return [past_points[-1][0] + lat_diff * time_diff, past_points[-1][1] + lon_diff * time_diff, past_points[-1][2] + alt_diff * time_diff]

def update_trajectories():
    while True:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        response = requests.get(f"{CONSTELLATION_ENDPOINT}00.json")
        print("Fetching latest data", response)
        if response.status_code == 200:
            try:
                data = json.loads(response.content.decode('utf8'))
                timestamp = datetime.now(timezone.utc).isoformat()
                for rowno, balloon in enumerate(data):
                    if all(value == 0 for value in balloon) or not all(isinstance(value, (int, float)) and not np.isnan(value) for value in balloon):
                        past_points = []
                        cursor.execute('''
                            SELECT latitude, longitude, altitude FROM trajectories 
                            WHERE rowno = ? ORDER BY timestamp DESC LIMIT 2
                        ''', (rowno,))
                        past_points = cursor.fetchall()
                        predicted_values = linear_extrapolate(past_points, 1)
                        if predicted_values:
                            lat, lon, alt = predicted_values
                            predicted_flag = 1
                        else:
                            continue
                    else:
                        lat, lon, alt = balloon
                        predicted_flag = 0
                    
                    found = False
                    cursor.execute('''
                        SELECT balloon_id, latitude, longitude FROM trajectories 
                        WHERE active = 1 AND rowno = ? ORDER BY timestamp DESC
                    ''', (rowno,))
                    candidates = cursor.fetchall()
                    for balloon_id, last_lat, last_lon in candidates:
                        # dist = haversine(last_lat, last_lon, lat, lon)
                        dist_lat = abs(last_lat - lat)
                        dist_lon = abs(last_lon - lon)
                        print(last_contacted - i, dist, balloon_id, _)
                        #print(dist)
                        if dist_lat < 3.5 and dist_lon < 3.5:
                            cursor.execute('''
                                INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (balloon_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag))
                            found = True
                            break
                    if not found:
                        new_id = str(uuid.uuid4())
                        cursor.execute('''
                            INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (new_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag))
                conn.commit()
            except Exception as e:
                print(f"Error while parsing: {e}")
        conn.close()
        time.sleep(3600)

def build_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    last_contacted = 25
    first_time = True
    for i in range(24, -1, -1):
        response = requests.get(f"{CONSTELLATION_ENDPOINT}{str(i).zfill(2)}.json")
        print(i, response)
        if response.status_code == 200:
            try:
                if response.content.decode('utf8').splitlines()[0] != '[':
                    data = '[\n'+response.content.decode('utf8')
                else:
                    data = response.content.decode('utf8')
                data = json.loads(data)
                timestamp = (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
                
                for rowno, balloon in enumerate(data):
                    print(f'rowno {rowno}')
                    found = False
                    lat, lon, alt = 0, 0, 0
                    if not first_time and (all(value == 0 for value in balloon) or not all(isinstance(value, (int, float)) and not np.isnan(value) for value in balloon)):
                        past_points = []
                        cursor.execute('''
                            SELECT latitude, longitude, altitude FROM trajectories 
                            WHERE rowno = ? ORDER BY timestamp DESC LIMIT 2
                        ''', (rowno,))
                        past_points = cursor.fetchall()
                        
                        try:
                            predicted_values = linear_extrapolate(past_points, last_contacted - i)
                        except Exception as e:
                            print(f"linear extrapolation not possible, not enough data points ",past_points)
                        if predicted_values:
                            lat, lon, alt = predicted_values
                            predicted_flag = 1
                        else:
                            continue
                    else:
                        lat, lon, alt = balloon
                        predicted_flag = 0                                 

                    # https://chatgpt.com/share/67b0ddda-6a7c-8013-bdcb-cec997cb853e
                    # interesting line of prompts could be useful later
                    cursor.execute('''
                        SELECT balloon_id, latitude, longitude, rowno, timestamp FROM trajectories 
                        WHERE active = 1 AND rowno = ? AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? AND corrupted = ? AND (possible_kin is NULL OR possible_kin = '')
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (rowno, lat - QUADRANT_SIZE, lat + QUADRANT_SIZE, lon - QUADRANT_SIZE, lon + QUADRANT_SIZE, 0))
                    candidates = cursor.fetchall()
                    print(f'prime candidates: {candidates}')
                   
                    # they could probably be associated with any other row's trajectory. coz every row need not map to the same row in the next hour
                    if not candidates:
                        cursor.execute('''
                            SELECT balloon_id, latitude, longitude, rowno, timestamp FROM trajectories 
                            WHERE active = 1 AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? AND corrupted = ? AND (possible_kin is NULL OR possible_kin = '')
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (lat - QUADRANT_SIZE, lat + QUADRANT_SIZE, lon - QUADRANT_SIZE, lon + QUADRANT_SIZE, 0))
                        candidates = cursor.fetchall()  

                    print(candidates)
                    for balloon_id, last_lat, last_lon,_, time in candidates:
                        if first_time: break
                        dist_lat, dist_lon = [2 * (last_contacted - i)] * 2 
                        try:
                            
                            if isinstance(last_lat, (int, float)) and isinstance(last_lon, (int, float)):
                                dist = haversine(last_lat, last_lon, lat, lon)
                                #print(f'dist = {dist}')
                                dist_lat = abs(last_lat - lat)
                                dist_lon = abs(last_lon - lon)
                                # print(last_contacted - i, dist, balloon_id, _)
                                #print(dist)
                            else:
                                #dist = 60
                                dist_lat, dist_lon = [2 * (last_contacted - i)] * 2 # just a dummy above the threshold
                        except Exception as e:
                            print(f"haversine failing, {last_lat}, {last_lon}, {lat}, {lon}, {candidates}")
                        
                        if dist_lat < 1.9 * (last_contacted - i) and dist_lon < 1.9 * (last_contacted - i):
                            try:
                                cursor.execute('''
                                    INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, ass_rowno, d_moved)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (balloon_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag, _, dist))
                                found = True
                                break

                            except Exception as e:
                                print(f"exception in db insertion: {e}")
                                print(f"values parsed: {lat}, {lon}, {alt}")
                    
                    if not found:
                        try:
                            corrupted = 0
                            possible_kin = ''
                            strike = 0
                            new_id = str(uuid.uuid4())
                            print(first_time)

                            
                            if not first_time:
                                print('not firsttime')
                                # create redundancy with 3 strikes
                                # this is basically my way of modelling data corruption -- i am going to have a system that basically associates any lat,lon change to both its "possible" candidate and a new trajectory. if after 3 strikes it continues to be associated with a new traj rather than its potential candiate, we let it be a new set of trajectory and deactivate the old trajectory. On the flip side if we get it back on track with its candidate (this what we call data corruption here), we mark the corrupted data, place predicted data in its place (show both actually) and then flag the newly created bucket (not the same as deactivation).
                                cursor.execute('''
                                    SELECT latitude, longitude, altitude, balloon_id, rowno, strike FROM trajectories 
                                    WHERE active = 1 AND rowno = ? AND corrupted = ? AND (possible_kin IS NULL OR possible_kin = '')
                                    ORDER BY timestamp DESC LIMIT 1
                                ''', (rowno, 0)) 
                                candidates = cursor.fetchall()
                                possible_kin = candidates[0][3]
                                dist = haversine(candidates[0][0], candidates[0][1], lat, lon)

                                corrupted = 1
                                strike = max(1, int(candidates[0][5]) + 1)
                            
                                if strike <= 3:
                                    try:
                                        p_lat, p_lon, p_alt = linear_extrapolate(past_points, last_contacted - i)
                                        cursor.execute('''
                                            INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, corrupted, strike, d_moved)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (possible_kin, rowno, p_lat, p_lon, p_alt, timestamp, 1, 1, 0, strike, dist))

                                    except Exception as e:
                                        print(f"here.. linear extrapolation not possible, not enough data points ",past_points)
                                    cursor.execute('''
                                        INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, corrupted, strike, d_moved)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (possible_kin, rowno, lat, lon, alt, timestamp, 1, 0, corrupted, strike, dist))
                                    
                                    past_points = candidates
                                    
    
    
                                else:
                                    possible_kin = ''
                                    # YOU HAVE TO DEACTIVATE YOUR OLD TRAJECTORIES HERE
                             
                            cursor.execute('''
                                SELECT latitude, longitude, altitude, balloon_id, rowno, strike FROM trajectories 
                                WHERE active = 1 AND rowno = ? AND possible_kin = ?
                                ORDER BY timestamp DESC LIMIT 1
                            ''', (rowno, possible_kin)) 
                            candidates = cursor.fetchall()
                            
                            if candidates:
                                new_id = candidates[0][3]
                            

                            cursor.execute('''
                                INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, possible_kin)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (new_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag, possible_kin))

                        except Exception as e:
                            print(f"db insertion exception: {e}")
                            print(f"values parsed: {lat}, {lon}, {alt}")
                last_contacted = i
                first_time = False
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                line_number = exc_traceback.tb_lineno
                print(f"Error: {e} on line {line_number}")
    
    conn.commit()
    conn.close()

app = Flask(__name__)
CORS(app)
PLOT_DIR = "trajectory_plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def get_all_trajectories():
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    
    # Get all unique balloon IDs
    cursor.execute("SELECT DISTINCT balloon_id FROM trajectories WHERE active = 1")
    balloon_ids = [row[0] for row in cursor.fetchall()]
    
    trajectories = []
    for balloon_id in balloon_ids:
        cursor.execute("""
            SELECT latitude, longitude, altitude, timestamp, predicted, rowno
            FROM trajectories 
            WHERE balloon_id = ? 
            ORDER BY timestamp
        """, (balloon_id,))
        
        points = cursor.fetchall()
        if points:
            trajectory = {
                'balloon_id': balloon_id,
                'points': [{
                    'lat': point[0],
                    'lng': point[1],
                    'altitude': point[2],
                    'timestamp': point[3],
                    'predicted': bool(point[4]),
                    'rowno': point[5]
                } for point in points]
            }
            trajectories.append(trajectory)
    
    conn.close()
    return trajectories

def generate_3d_plot(balloon_id):
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT latitude, longitude, altitude, predicted, corrupted, timestamp
        FROM trajectories 
        WHERE balloon_id = ? 
        ORDER BY timestamp
    """, (balloon_id,))
    
    points = cursor.fetchall()
    conn.close()
    

    
    if not points or len(points) < 3:
        return None
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate regular and predicted points
    regular_points = [(p[0], p[1], p[2]) for p in points if not p[3] and not p[4]]
    predicted_points = [(p[0], p[1], p[2]) for p in points if p[3]]
    corrupted_points = [(p[0], p[1], p[2]) for p in points if p[4]]
    
    if regular_points:
        lats, lons, alts = zip(*regular_points)
        ax.scatter(lons, lats, alts, c='blue', marker='o', label='Actual Points')
        ax.plot(lons, lats, alts, 'b-', alpha=0.5)
    
    if predicted_points:
        lats, lons, alts = zip(*predicted_points)
        ax.scatter(lons, lats, alts, c='green', marker='^', label='Predicted Points')

    if corrupted_points:
        lats, lons, alts = zip(*corrupted_points)
        ax.scatter(lons, lats, alts, c='red', marker='x', label='Corrupted Points')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    ax.set_title(f'Balloon Trajectory {balloon_id}')
    ax.legend()
    
    plot_path = os.path.join(PLOT_DIR, f'trajectory_{balloon_id}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

@app.route('/api/trajectories', methods=['GET'])
def get_trajectories():
    return jsonify(get_all_trajectories())

@app.route('/api/plot/<balloon_id>', methods=['GET'])
def get_plot(balloon_id):
    plot_path = generate_3d_plot(balloon_id)
    if plot_path:
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': 'Trajectory not found'}), 404

def visualize():
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    
    # Get all unique balloon IDs
    cursor.execute("SELECT DISTINCT balloon_id FROM trajectories WHERE active = 1")
    balloon_ids = [row[0] for row in cursor.fetchall()]
    
    trajectories = []
    for balloon_id in balloon_ids:
        generate_3d_plot(balloon_id)

if __name__ == '__main__':
    print(os.path.isfile(DB_FILE))
    if os.path.isfile(DB_FILE):
        init_db()
        visualize()
    else:
        init_db()
        build_history()
        visualize()
    #app.run(debug=True, port=5000)
