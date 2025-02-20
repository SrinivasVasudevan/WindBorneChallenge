import sqlite3
import sys
import requests
import json
import time
import uuid
import random
import os
from datetime import datetime, timedelta, timezone
from filterpy.kalman import KalmanFilter
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from flask_cors import CORS
from threading import Thread
import openmeteo_requests
import math

from retry_requests import retry
import requests_cache


import threading
import matplotlib.pyplot as plt



############################

# no_requests = 0

WIND_API = "https://api.open-meteo.com/v1/forecast"
API_LIMIT_FILE = "api_usage.json"
DAILY_API_LIMIT = 5000  # Set your daily limit here

trajectories = []

def load_api_usage():
    try:
        with open(API_LIMIT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"count": 0}

def save_api_usage(count):
    with open(API_LIMIT_FILE, "w") as f:
        json.dump({"count": count}, f)

def check_api_limit():
    usage = load_api_usage()
    return usage["count"] < DAILY_API_LIMIT

def increment_api_usage():
    usage = load_api_usage()
    usage["count"] += 1
    save_api_usage(usage["count"])


####################


def get_new_coordinates(lat1, lon1, distance_km, bearing_degrees):
    R = 6371  
    lat1, lon1, bearing = map(math.radians, [lat1, lon1, bearing_degrees])

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing))
    
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

def predict_traj(balloon_id, cursor, windPresent = False):
    history = get_trajectory_from_db(balloon_id, cursor)
    if not history:
        return jsonify({"error": "Balloon not found"}), 404
    
    lat,lon,alt = history[0]["lat"], history[0]["lng"], history[0]["altitude"]
    print('gonna predict for this:: ',lat, lon, alt, history[0]["timestamp"], list(reversed(history)), )
    gpr_pred = predict_using_gpr(list(reversed(history)))
    if windPresent:
        wind_speed, wind_dir = get_wind_data(lat, lon, alt)

    predicted_lat1, predicted_lon1, predicted_alt1 = [None]*3
    predicted_lat2, predicted_lon2 = [None]*2

    predicted_lat, predicted_lon, predicted_alt = None, None, None

    if gpr_pred:
        predicted_lat1, predicted_lon1, predicted_alt1 = gpr_pred
    elif len(history) >= 2:
        predicted_lat1, predicted_lon1, predicted_alt1 = linear_extrapolate([[history[0]["lat"], history[0]["lng"], history[0]["altitude"]],[history[-1]["lat"], history[-1]["lng"], history[-1]["altitude"]]])

    if windPresent and wind_speed is not None and wind_dir is not None:
        predicted_lat2, predicted_lon2 = get_new_coordinates(lat, lon, wind_speed, wind_dir)
    
    if predicted_lat1 and predicted_lon1 and predicted_alt1:
        if predicted_lat2 and predicted_lon2:
            predicted_lat = predicted_lat1 * 0.5 + predicted_lat2 * 0.5
            predicted_lon = predicted_lon1 * 0.5 + predicted_lon2 * 0.5
            predicted_alt = predicted_alt1
        else:
            predicted_lat = predicted_lat1
            predicted_lon = predicted_lon1
    elif predicted_lat2 and predicted_lon2:
        predicted_lat = predicted_lat2
        predicted_lon = predicted_lon2
    
    if predicted_lat and predicted_lon:
        if predicted_alt:
            return [predicted_lat, predicted_lon, predicted_alt]
        else:
            return [predicted_lat, predicted_lon, alt]
        
    else:
        return [None, None, None]



CONSTELLATION_ENDPOINT = "https://a.windbornesystems.com/treasure/"
TRAJECTORY_DATA = "https://a.windbornesystems.com/treasure/00.json" # new ones are here
DATA_DIR = "balloon_data"
DB_FILE = 'balloon_trajectories.db'
DISTANCE_THRESHOLD_KM = 50  # based on this old https://www.weather.gov/media/key/KEY%20-%20Weather%20Balloon%20Poster.pdf data and https://sites.wff.nasa.gov/code820/pages/about/about-faq.html#:~:text=The%20balloon%20typically%20rises%20at,120%2C000%20ft%20(36.6%20km).
# will change this to be dyanmic on history later
QUADRANT_SIZE = 10  # Define quadrant size for spatial partitioning

app = Flask(__name__, static_folder="../front-end/build", static_url_path='/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

#socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})
PLOT_DIR = "trajectory_plots"

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
            d_moved REAL,
            mv_median REAL
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

def linear_extrapolate(past_points):
    if len(past_points) < 2:
        return None
    print(past_points)
    lat_diff = past_points[-1][0] - past_points[-2][0]
    lon_diff = past_points[-1][1] - past_points[-2][1]
    alt_diff = past_points[-1][2] - past_points[-2][2]
    print([past_points[-1][0] + lat_diff, past_points[-1][1] + lon_diff, past_points[-1][2] + alt_diff])
    return [past_points[-1][0] + lat_diff, past_points[-1][1] + lon_diff, past_points[-1][2] + alt_diff]

def parse_response(response, cursor, call_time = 0, first_time = False):
    if response.content.decode('utf8').splitlines()[0] != '[':
        data = '[\n'+response.content.decode('utf8')
    else:
        data = response.content.decode('utf8')

    data = json.loads(data)
    
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=call_time)).isoformat() 
    print(timestamp)   
    
    for rowno, balloon in enumerate(data):
        found = False
        
        lat, lon, alt = 0, 0, 0

        # curb noise here
        if not first_time and (all(value == 0 for value in balloon) or not all(isinstance(value, (int, float)) and not np.isnan(value) for value in balloon)):
            past_points = []
            cursor.execute('''
                SELECT latitude, longitude, altitude, timestamp, balloon_id FROM trajectories 
                WHERE rowno = ? AND (possible_kin is NULL OR possible_kin = '') ORDER BY timestamp DESC LIMIT 2
            ''', (rowno,))
            past_points = cursor.fetchall()
            
            try:
                if past_points and len(past_points) >= 2:
                    predicted_values = predict_traj(past_points[0][4], cursor)
                    if predicted_values:
                        last_time = past_points[0][3]
                        time_diff = ((datetime.now(timezone.utc) - timedelta(hours=call_time))-datetime.fromisoformat(last_time)).total_seconds()/3600
                        time_diff = abs(time_diff)
                        lat, lon, alt = [val for val in predicted_values]
                        predicted_flag = 1
                    else:
                        continue
            except Exception as e:
                print(f"linear extrapolation not possible, not enough data points ",past_points)

            
        else:
            lat, lon, alt = balloon
            predicted_flag = 0                                 

        # https://chatgpt.com/share/67b0ddda-6a7c-8013-bdcb-cec997cb853e
        # interesting line of prompts could be useful later
        if not lat and not lon and not alt: continue
        mm = 0 
        alt_candidates = False
        curr_id = None
        c_time = None

        cursor.execute('''
            SELECT balloon_id, latitude, longitude, rowno, timestamp, d_moved FROM trajectories 
            WHERE active = 1 AND rowno = ? AND corrupted = ? AND (possible_kin is NULL OR possible_kin = '')
            ORDER BY timestamp DESC LIMIT 6
        ''', (rowno, 0))
        candidates = cursor.fetchall()

        try:
            if not first_time and candidates:
                balloon_id, last_lat, last_lon,_, time, d_moved  = candidates[0]
                c_time = time
                dist = None
                if isinstance(last_lat, (int, float)) and isinstance(last_lon, (int, float)):
                    dist = haversine(last_lat, last_lon, lat, lon)
                    if dist == 0: continue
                    time_diff = ((datetime.now(timezone.utc) - timedelta(hours=call_time))-datetime.fromisoformat(time)).total_seconds()/3600
                    time_diff = abs(time_diff)
                    print('tdiff: ', time_diff)
                    print(len(candidates), rowno)
                    if len(candidates) == 1:
                        mm = dist / time_diff
                        curr_id = balloon_id
                    else:    
                        d_list = [(candidate[5], candidate[0]) for candidate in candidates if candidate[5]]
                        d_list = sorted(d_list, key=lambda tup: tup[0])
                        if len(d_list) > 0 and len(d_list)%2==0:
                            mm = (d_list[len(d_list)//2][0] + d_list[len(d_list)//2-1][0])//2
                            curr_id = d_list[len(d_list)//2 - random.randint(0, 1)][1]
                        elif len(d_list) > 0:
                            mm = d_list[len(d_list)//2][0]
                            curr_id = d_list[len(d_list)//2][1]

                    # 50 again is based on the distance threshold articles
                    print(dist, mm)
                    if not mm or not dist or dist / time_diff > (mm + 50) or dist / time_diff < (mm - 50):
                        candidates = None

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            line_number = exc_traceback.tb_lineno
            print(f"Error: {e} on line {line_number}")  


        # they could probably be associated with any other row's trajectory. coz every row need not map to the same row in the next hour
        if not candidates:
            cursor.execute('''
                SELECT balloon_id, latitude, longitude, rowno, timestamp, d_moved FROM trajectories 
                WHERE active = 1 AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? AND corrupted = ? AND  (possible_kin is NULL OR possible_kin = '')
                ORDER BY timestamp DESC LIMIT 1
            ''', (lat - 2, lat + 2, lon - 2, lon + 2, 0))
            candidates = cursor.fetchall()  
            if candidates:
                curr_id = candidates[0][0]
                c_time = candidates[0][4]

        if c_time:
            time_diff = ((datetime.now(timezone.utc) - timedelta(hours=call_time))-datetime.fromisoformat(c_time)).total_seconds()/3600
            time_diff = abs(time_diff)
            print(time_diff)

        if not first_time and curr_id and time_diff and time_diff >= 0.5:
            cursor.execute('''
                INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, ass_rowno, d_moved, mv_median)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (curr_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag, _, dist / time_diff, mm))
            found = True
            

        if not found:
            try:
                corrupted = 0
                possible_kin = ''
                strike = 0
                new_id = str(uuid.uuid4())
                cursor.execute('''
                        SELECT latitude, longitude, altitude, balloon_id, rowno, strike, timestamp FROM trajectories 
                        WHERE active = 1 AND rowno = ? AND (possible_kin IS NULL OR possible_kin = '')
                        ORDER BY timestamp DESC LIMIT 2
                    ''', (rowno, )) 
                candidates = cursor.fetchall()
                if not first_time and candidates:
                    # create redundancy with 3 strikes
                    # this is basically my way of modelling data corruption -- i am going to have a system that basically associates any lat,lon change to both its "possible" candidate and a new trajectory. if after 3 strikes it continues to be associated with a new traj rather than its potential candiate, we let it be a new set of trajectory and deactivate the old trajectory. On the flip side if we get it back on track with its candidate (this what we call data corruption here), we mark the corrupted data, place predicted data in its place (show both actually) and then flag the newly created bucket (not the same as deactivation).
                    
                    
                    possible_kin = candidates[0][3]
                    last_time = candidates[0][6]
                    dist = haversine(candidates[0][0], candidates[0][1], lat, lon)

                    corrupted = 1
                    strike = max(1, int(candidates[0][5]) + 1)
                
                    if strike <= 3:
                        try:
                            time_diff = ((datetime.now(timezone.utc) - timedelta(hours=call_time))-datetime.fromisoformat(last_time)).total_seconds()/3600
                            time_diff = abs(time_diff)
                            #print('candidates for prediction: ',len(candidates), candidates[0][3])
                            p_lat, p_lon, p_alt = predict_traj(candidates[0][3], cursor)
                            if p_lat and p_lon and p_lat:
                                
                                cursor.execute('''
                                    INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, corrupted, strike, d_moved)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (possible_kin, rowno, p_lat, p_lon, p_alt, timestamp, 1, 1, 0, strike, dist / time_diff))

                        except Exception as e:
                            print(f"here.. linear extrapolation not possible, not enough data points ",candidates)
                        
                        cursor.execute('''
                            INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, corrupted, strike, d_moved)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (possible_kin, rowno, lat, lon, alt, timestamp, 1, 0, corrupted, strike, dist / time_diff))

                    else:
                        possible_kin = ''
                        # YOU HAVE TO DEACTIVATE YOUR OLD TRAJECTORIES HERE
                    
                cursor.execute('''
                    SELECT latitude, longitude, altitude, balloon_id, rowno, strike, timestamp FROM trajectories 
                    WHERE active = 1 AND rowno = ? AND possible_kin = ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (rowno, possible_kin)) 
                candidates = cursor.fetchall()
                
                dist = 0
                time_diff = 1

                if candidates:
                    dist = haversine(last_lat, last_lon, lat, lon)
                    last_time = candidates[0][6]
                    time_diff = ((datetime.now(timezone.utc) - timedelta(hours=call_time))-datetime.fromisoformat(last_time)).total_seconds()/3600
                    time_diff = abs(time_diff)
                    new_id = candidates[0][3]

                cursor.execute('''
                    INSERT INTO trajectories (balloon_id, rowno, latitude, longitude, altitude, timestamp, active, predicted, possible_kin, d_moved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (new_id, rowno, lat, lon, alt, timestamp, 1, predicted_flag, possible_kin, dist / time_diff))

            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                line_number = exc_traceback.tb_lineno
                print(f"db insertion exception: {e} {line_number}")
                print(f"values parsed: {lat}, {lon}, {alt}")

def update_trajectories():
    while True:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        response = requests.get(f"{CONSTELLATION_ENDPOINT}00.json")
        print("Fetching latest data", response)
        if response.status_code == 200:
            try:
                parse_response(response, cursor)
            except Exception as e:
                print(f"Error while parsing: {e}")

        conn.commit()
        conn.close()
        time.sleep(1500)


def build_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    #last_contacted = 25
    first_time = True
    for i in range(24, -1, -1):
        response = requests.get(f"{CONSTELLATION_ENDPOINT}{str(i).zfill(2)}.json")
        print(i, response)
        if response.status_code == 200:
            try:
                parse_response(response, cursor, i, first_time)
                first_time = False
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                line_number = exc_traceback.tb_lineno
                print(f"Error: {e} on line {line_number}")
    
    conn.commit()
    conn.close()



if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def get_all_trajectories():
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    
    # Get all unique balloon IDs
    cursor.execute("""
        SELECT DISTINCT balloon_id FROM trajectories
        WHERE active = 1 
        AND balloon_id IN (
            SELECT balloon_id FROM trajectories
            GROUP BY balloon_id
            HAVING COUNT(*) > 3
        )
    """)
    
    balloon_ids = [row[0] for row in cursor.fetchall()]
    
    trajectories = []
    for balloon_id in balloon_ids:
        cursor.execute("""
            SELECT latitude, longitude, altitude, timestamp, predicted, rowno, corrupted
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
                    'rowno': point[5],
                    'corrupted': point[6]
                } for point in points]
            }
            trajectories.append(trajectory)
    
    conn.close()
    print(trajectories)
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
    

    
    if not points or len(points) < 4:
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
    print(jsonify(get_all_trajectories()))
    return jsonify(get_all_trajectories())

@app.route('/api/plot/<balloon_id>', methods=['GET'])
def get_plot(balloon_id):
    plot_path = generate_3d_plot(balloon_id)
    if plot_path:
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': 'Trajectory not found'}), 404

def get_wind_data(lat, lon, alt):
    if not check_api_limit():
        return None, None
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["wind_speed_10m", "wind_direction_10m"]
        }
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        response = openmeteo.weather_api(WIND_API, params=params)
        response = response[0]
        print('response from windapi', response)
        if response:
            current = response.Current()
            current_wind_speed_10m = current.Variables(0).Value()
            current_wind_direction_10m = current.Variables(1).Value()
            increment_api_usage()
            return current_wind_speed_10m, current_wind_direction_10m
        return None, None
    except Exception as e:
        print(f"Error fetching wind data: {e}")
        return None, None

def get_trajectory_from_db(balloon_id, cursor):
    # conn = sqlite3.connect(DB_FILE)
    # cursor = conn.cursor()
    try:
        cursor.execute("SELECT latitude, longitude, altitude, timestamp FROM trajectories WHERE balloon_id = ? AND predicted = 0 AND corrupted = 0 AND (possible_kin is NULL OR possible_kin = '') ORDER BY timestamp DESC LIMIT 3", (balloon_id,))
        rows = cursor.fetchall()
        # conn.close()
        print('rows fetched: ', rows)
        if not rows:
            return None
        return [{"lat": r[0], "lng": r[1], "altitude": r[2], "timestamp": abs((datetime.fromisoformat(r[3]) - datetime.now(timezone.utc)).total_seconds()/3600)} for r in rows]
    except Exception as e:
        print(e)

def predict_using_gpr(history):
    try:
        if len(history) < 2:
            return None
        X = np.array([[p["timestamp"]] for p in history])
        y_lat = np.array([p["lat"] for p in history]).reshape(-1, 1)
        y_lon = np.array([p["lng"] for p in history]).reshape(-1, 1)
        y_alt = np.array([p["altitude"] for p in history]).reshape(-1, 1)
        scaler_X = StandardScaler()
        scaler_lat = StandardScaler()
        scaler_lon = StandardScaler()
        scaler_alt = StandardScaler()

       
        X_scaled = scaler_X.fit_transform(X)
        y_lat_scaled = scaler_lat.fit_transform(y_lat)
        y_lon_scaled = scaler_lon.fit_transform(y_lon)
        y_alt_scaled = scaler_alt.fit_transform(y_alt)

        kernel = C(1.0) * RBF(1.0)
        gpr_lat = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr_lon = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr_alt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        gpr_lat.fit(X_scaled, y_lat_scaled.ravel())
        gpr_lon.fit(X_scaled, y_lon_scaled.ravel())
        gpr_alt.fit(X_scaled, y_alt_scaled.ravel())

        future_time_scaled = scaler_X.transform([[X[-1, 0] - 1]])
        pred_lat_scaled = gpr_lat.predict(future_time_scaled)[0]
        pred_lon_scaled = gpr_lon.predict(future_time_scaled)[0]
        pred_alt_scaled = gpr_alt.predict(future_time_scaled)[0]

        pred_lat = scaler_lat.inverse_transform([[pred_lat_scaled]])[0, 0]
        pred_lon = scaler_lon.inverse_transform([[pred_lon_scaled]])[0, 0]
        pred_alt = scaler_alt.inverse_transform([[pred_alt_scaled]])[0, 0]

        print(pred_lat, pred_lon, pred_alt, "history: ", history, [X[-1, 0] - 1])
    except Exception as e:
        print(e)

    return pred_lat, pred_lon, pred_alt
        
        
        

@app.route("/api/predict/<balloon_id>", methods=["GET"])
def predict_trajectory(balloon_id):
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    pred_lat, pred_lon, pred_alt = predict_traj(balloon_id,cursor, True)
    if pred_lat == None or pred_lon == None or pred_alt == None:
        return {}
    
    new_point = {
        "lat": pred_lat,
        "lng": pred_lon,
        "altitude": pred_alt,
        "timestamp": "Predicted"
    }
    print(new_point)
    conn.commit()
    conn.close()

    return jsonify([new_point])

def visualize():
    conn = sqlite3.connect('balloon_trajectories.db')
    cursor = conn.cursor()
    
    # Get all unique balloon IDs
    cursor.execute("SELECT DISTINCT balloon_id FROM trajectories WHERE active = 1")
    balloon_ids = [row[0] for row in cursor.fetchall()]
    
    trajectories = []
    for balloon_id in balloon_ids:
        generate_3d_plot(balloon_id)

print(os.path.isfile(DB_FILE))
if os.path.isfile(DB_FILE):
    init_db()
    #visualize()
else:
    init_db()
    build_history()
    #visualize()
thread = Thread(target=update_trajectories)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    app.run()
