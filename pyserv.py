from flask import Flask, request, after_this_request, jsonify
import json
import numpy as np
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from tensorflow.keras import backend as K

from flask_cors import CORS, cross_origin


features = ["Height", "conf", "Role", "team",  "Min_per%","PRPG!","D-PRPG","BPM","OBPM","DBPM","Ortg","D-Rtg","usg","eFG","TS_per","ORB_per","DRB_per","AST_per","TO_per","Blk","Stl","FTR","Dunks%","Close 2 %","Far 2 %","FT_per","2P%","3PR","3P%","Ast","Reb","Pts","DunksA","DunksM","Close2A","Close2M","Far2A","Far2M","FTA","FTM","2PA","2PM","3PA","3PM"]

data_path = 'final_data.csv'  # Replace with your actual dataset file
ball_data = pd.read_csv(data_path, low_memory=False)

ball_data.columns = ball_data.columns.str.strip()  
testData = ball_data.loc[ball_data['year'] < 2024]
testData = ball_data.loc[ball_data['Pick'] > 0]

dclass = ball_data.loc[ball_data['year'] == 2024]
resultData = dclass[features]

# Define features and target

X = testData[features]
y = testData['Pick']

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

bagging_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=50,
    random_state=0
)
bagging_model.fit(train_X, train_y)

data_path = 'final_data.csv'  # Replace with your actual dataset file
ball_data = pd.read_csv(data_path, low_memory=False)
ball_data['drafted'] = ball_data['Pick'].apply(lambda x: 1 if x > 0 else 0)

ball_data.columns = ball_data.columns.str.strip()  
testData = ball_data.loc[ball_data['year'] < 2024]

resultData = ball_data.loc[ball_data['year'] == 2024]
resultData = resultData[features]

# Define features and target
X = testData[features]
y = testData['drafted']

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the first neural network model using MAE
nn_model = Sequential([
    Dense(128, activation='relu', input_dim=train_X.shape[1]),  # Input layer
    Dropout(0.2),  # Dropout for regularization
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification (sigmoid activation)
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the first model (using binary cross-entropy with accuracy metric)
nn_model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

#To endure consistent clusteri  ng
np.random.seed(42)

#Dictionary to store fuzzifiers, centroids, and k 
profiles = {
    'shooter': {
        'fuzzifier': 1.5,
        'k': 4,
        'centroids': None,
        'stats': ["Far2A_pg", "Close2A_pg", "FTA_pg", "OBPM", "eFG", "3PA_pg", "Far 2 %", "Close 2 %", "3P%"]
    },
    'slasher': {
        'fuzzifier': 2.0,
        'k': 3,
        'centroids': None,
        'stats': ["Height", "DunksA_pg", 'Close2A_pg', 'FTA_pg', 'usg', 'ORB_per', "Ast"]
    },
    'playmaker': {
        'fuzzifier': 1.6,
        'k': 4,
        'centroids': None,
        'stats': ["Ast", "TO_per", "3P%", "PRPG!", "Ortg", "TS_per", "Stl", "eFG", "FT_per", "FTA_pg"]
    },
    'rebounding': {
        'fuzzifier': 2.0,
        'k': 3,
        'centroids': None,
        'stats': ["ORB_per", "DunksA", "FC/40", "Ortg", "DRB_per", "Blk", "D-Rtg", "AST_per"]
    },
    'defense': {
        'fuzzifier': 2.0,
        'k': 4, 
        'centroids': None,
        'stats': ["Height", "BPM","D-PRPG", "DBPM", "D-Rtg", "DRB_per", "Blk", "Stl", 'DRB_per']
    }
}

def cluster_analysis(data, profile_name):
    profile = profiles[profile_name]
    stats_list = profile['stats']
    fuzzifier = profile['fuzzifier']
    k = profile['k']  # Use the optimal k for the profile
    
    data_for_clustering = data[stats_list].copy()
    
    #Fill missing values 
    data_for_clustering.fillna(data_for_clustering.mean(), inplace=True)
    
    #Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    data_transposed = scaled_data.T

    #Fuzzy c-means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_transposed, c=k, m=fuzzifier, error=0.005, maxiter=1000, init=None
    )

    #Store the centroids in profile to use laterr
    profile['centroids'] = cntr

    #Add membership values and fuzzy cluster labels
    for i in range(k):
        data[f'{profile_name}_membership_{i}'] = u[i, :]
    data[f'{profile_name}_fuzzy_cluster'] = np.argmax(u, axis=0)

    #Save clustering results to csv
    data.to_csv(f"{profile_name}_clusters.csv", index=False)

    return data

# Predict clustering of a new player
def predict_cluster(profile_name, input_stats):
    profile = profiles[profile_name]

    #Centroids and fuzzifier for the selected profile
    centroids = profile['centroids']
    fuzzifier = profile['fuzzifier']

    #Centroid not here, try clustering

    #Scalte input
    input_stats_scaled = np.array(input_stats).reshape(1, -1)
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_stats_scaled)

    #Distances to centroids and fuzzy membership
    distances = np.linalg.norm(centroids - input_scaled, axis=1)
    membership_values = 1 / (1 + (distances / np.min(distances))**(2 / (fuzzifier - 1)))

    membership_values /= np.sum(membership_values)
    return membership_values, np.argmax(membership_values)

#Import data and sort onlt drafted players
df = pd.read_csv("final_data.csv")
df = df.sort_values(by='Pick')
year = 2010
df = df[df['year'] >= year]
df = df[df['Pick'] >= 1]


app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

# Define the feature names (ensure these match the order of the features in your model)
features = ["Height", "conf", "Role", "team", "Min_per%", "PRPG!", "D-PRPG", "BPM", "OBPM", "DBPM", "Ortg", "D-Rtg", 
            "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "Blk", "Stl", "FTR", "Dunks%", "Close 2 %", 
            "Far 2 %", "FT_per", "2P%", "3PR", "3P%", "Ast", "Reb", "Pts", "DunksA", "DunksM", "Close2A", "Close2M", "Far2A", 
            "Far2M", "FTA", "FTM", "2PA", "2PM", "3PA", "3PM"]

position_map = {'Wing F': 1, 'Stretch 4': 2, 'Combo G': 3, 'Scoring PG': 4, 'PF/C': 5, 'Wing G': 6, 'Pure PG': 7, 'C': 8}
grade_map = {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}
conf_map = {'Sum': 1, 'AE': 2, 'BE': 3, 'BSth': 4, 'MAAC': 5, 'OVC': 6, 'CUSA': 7, 'MAC': 8, 'NEC': 9, 'MVC': 10, 'SWAC': 11, 'BW': 12, 'BSky': 13, 'MWC': 14, 'Pat': 15, 'B10': 16, 'WCC': 17, 'A10': 18, 'Ivy': 19, 'Horz': 20, 'WAC': 21, 'Amer': 22, 'SB': 23, 'CAA': 24, 'MEAC': 25, 'B12': 26, 'ACC': 27, 'ASun': 28, 'SC': 29, 'Slnd': 30, 'SEC': 31, 'ind': 32, 'P12': 33, 'Ind': 34, 'GWC': 35}
team_map = {'Denver': 1, 'NJIT': 2, 'Seton Hall': 3, 'Charleston Southern': 4, 'Rider': 5, 'Little Rock': 6, 'Marist': 7, 'Western Kentucky': 8, 'Middle Tennessee': 9, 'Western Michigan': 10, 'Northern Illinois': 11, 'Wagner': 12, 'Evansville': 13, 'Jackson St.': 14, 'UC Irvine': 15, 'Sacramento St.': 16, 'Lindenwood': 17, 'UNLV': 18, 'Navy': 19, 'Minnesota': 20, 'San Diego': 21, 'Richmond': 22, 'Columbia': 23, 'Detroit Mercy': 24, 'Fairleigh Dickinson': 25, 'UC Riverside': 26, 'Chicago St.': 27, 'Utah Tech': 28, 'Tarleton St.': 29, 'Northern Kentucky': 30, 'New Hampshire': 31, 'Harvard': 32, 'Southern': 33, 'Cal St. Bakersfield': 34, 'Bucknell': 35, 'Loyola Marymount': 36, 'Temple': 37, 'Northern Colorado': 38, 'Georgia Southern': 39, 'Stony Brook': 40, 'Canisius': 41, 'Coppin St.': 42, 'Merrimack': 43, 'Niagara': 44, 'West Virginia': 45, 'Florida Atlantic': 46, 'Maine': 47, 'Eastern Michigan': 48, 'UCLA': 49, 'UNC Asheville': 50, 'Alcorn St.': 51, 'Texas Southern': 52, 'North Dakota': 53, 'Virginia': 54, 'Eastern Illinois': 55, 'North Texas': 56, 'Delaware': 57, 'Oakland': 58, 'Creighton': 59, 'Tennessee Martin': 60, 'West Georgia': 61, 'Quinnipiac': 62, 'Fairfield': 63, 'Bellarmine': 64, 'Massachusetts': 65, 'Maryland Eastern Sho': 66, 'Fresno St.': 67, 'Wake Forest': 68, 'Prairie View A&M': 69, 'Army': 70, 'Morgan St.': 71, 'Cleveland St.': 72, 'Louisiana Tech': 73, 'Illinois Chicago': 74, 'Northern Arizona': 75, 'Washington St.': 76, 'FIU': 77, 'LIU': 78, 'Iona': 79, 'South Carolina St.': 80, 'Tulane': 81, 'Chattanooga': 82, 'Central Arkansas': 83, 'Pacific': 84, 'Fordham': 85, 'San Francisco': 86, 'McNeese St.': 87, 'Pepperdine': 88, 'Alabama A&M': 89, 'Tennessee St.': 90, 'Arkansas Pine Bluff': 91, 'Kent St.': 92, 'Mississippi Valley S': 93, 'Coastal Carolina': 94, 'St. Bonaventure': 95, 'Campbell': 96, 'Western Carolina': 97, 'Dartmouth': 98, 'High Point': 99, 'Miami OH': 100, 'Robert Morris': 101, 'Belmont': 102, 'Eastern Washington': 103, 'Marshall': 104, 'Stephen F. Austin': 105, 'Indiana': 106, 'VCU': 107, 'UMKC': 108, 'Holy Cross': 109, "Mount St. Mary's": 110, 'Appalachian St.': 111, 'Sacred Heart': 112, 'Colorado St.': 113, 'Marquette': 114, 'Penn': 115, 'Wyoming': 116, 'Washington': 117, 'Saint Francis': 118, 'Liberty': 119, 'UTEP': 120, 'Lehigh': 121, 'Louisiana': 122, 'Weber St.': 123, 'Stetson': 124, 'Oklahoma': 125, 'UNC Greensboro': 126, 'Stanford': 127, 'The Citadel': 128, 'Incarnate Word': 129, 'Central Connecticut': 130, 'Monmouth': 131, 'Portland': 132, 'Lafayette': 133, 'Elon': 134, 'USC Upstate': 135, 'Georgetown': 136, 'Tulsa': 137, 'Nebraska Omaha': 138, 'Jacksonville': 139, 'Mercer': 140, 'Drake': 141, 'East Carolina': 142, 'VMI': 143, 'Houston Christian': 144, 'Presbyterian': 145, 'Eastern Kentucky': 146, 'Saint Louis': 147, 'Western Illinois': 148, 'Northwestern': 149, 'East Tennessee St.': 150, 'Vermont': 151, 'Southern Illinois': 152, 'Grambling St.': 153, 'Texas St.': 154, 'Idaho': 155, 'North Florida': 156, 'Cal St. Fullerton': 157, 'Boston University': 158, 'New Orleans': 159, 'Southeast Missouri S': 160, 'Texas A&M Corpus Chr': 161, 'Cal Poly': 162, 'Northern Iowa': 163, 'Albany': 164, 'Charlotte': 165, 'Georgia St.': 166, 'Loyola MD': 167, 'Southeastern Louisia': 168, 'Florida Gulf Coast': 169, 'Brown': 170, 'Abilene Christian': 171, 'UTSA': 172, 'Akron': 173, 'Old Dominion': 174, 'Bethune Cookman': 175, 'Louisiana Monroe': 176, 'UC Santa Barbara': 177, 'Winthrop': 178, 'California': 179, 'Notre Dame': 180, 'Mercyhurst': 181, 'Bowling Green': 182, 'Siena': 183, 'Youngstown St.': 184, 'Delaware St.': 185, 'Lamar': 186, 'LSU': 187, 'Utah St.': 188, 'Towson': 189, 'Montana': 190, 'Long Beach St.': 191, 'Idaho St.': 192, 'Villanova': 193, 'Duquesne': 194, 'Baylor': 195, 'South Carolina': 196, 'Furman': 197, 'Oral Roberts': 198, 'Davidson': 199, 'James Madison': 200, 'Kennesaw St.': 201, 'Florida A&M': 202, 'Ohio': 203, 'Queens': 204, 'South Dakota St.': 205, 'Morehead St.': 206, 'Rice': 207, 'USC': 208, 'Norfolk St.': 209, 'Nicholls St.': 210, 'UMBC': 211, 'Ball St.': 212, 'Wright St.': 213, 'UMass Lowell': 214, 'Missouri St.': 215, 'Louisville': 216, 'Drexel': 217, 'New Mexico': 218, 'Illinois St.': 219, 'UT Rio Grande Valley': 220, 'South Alabama': 221, 'Milwaukee': 222, 'Buffalo': 223, 'Radford': 224, 'Austin Peay': 225, 'Longwood': 226, 'Wichita St.': 227, 'Le Moyne': 228, 'Michigan': 229, 'Stonehill': 230, 'Manhattan': 231, 'UNC Wilmington': 232, 'Charleston': 233, 'Purdue Fort Wayne': 234, 'Hampton': 235, 'New Mexico St.': 236, 'Samford': 237, 'Mississippi': 238, 'Texas A&M Commerce': 239, 'UC San Diego': 240, 'Princeton': 241, 'American': 242, 'Colorado': 243, 'Wofford': 244, 'Grand Canyon': 245, 'Arkansas': 246, 'Northwestern St.': 247, 'Utah': 248, 'Northeastern': 249, 'Green Bay': 250, 'William & Mary': 251, 'North Carolina Centr': 252, 'Kentucky': 253, 'Utah Valley': 254, 'Air Force': 255, 'Lipscomb': 256, 'BYU': 257, 'Florida St.': 258, 'St. Thomas': 259, 'La Salle': 260, 'Southern Miss': 261, 'Murray St.': 262, 'Gardner Webb': 263, 'Troy': 264, 'Hawaii': 265, 'South Florida': 266, 'Indiana St.': 267, 'Houston': 268, 'Virginia Tech': 269, 'Cal St. Northridge': 270, 'Hofstra': 271, 'Sam Houston St.': 272, 'Kansas': 273, 'Binghamton': 274, 'San Jose St.': 275, 'Southern Indiana': 276, 'Colgate': 277, 'Iowa St.': 278, 'Texas A&M': 279, 'Missouri': 280, 'N.C. State': 281, 'George Mason': 282, 'Central Michigan': 283, 'Arizona': 284, 'Yale': 285, 'SIU Edwardsville': 286, 'Boston College': 287, 'Seattle': 288, "St. John's": 289, 'Memphis': 290, 'UAB': 291, 'Cornell': 292, 'Oregon St.': 293, 'Alabama St.': 294, "Saint Peter's": 295, 'Arizona St.': 296, 'Kansas St.': 297, 'Georgia': 298, 'Nebraska': 299, 'Mississippi St.': 300, 'Toledo': 301, 'Texas': 302, 'Illinois': 303, 'Xavier': 304, 'Arkansas St.': 305, 'Boise St.': 306, 'Providence': 307, 'Montana St.': 308, 'DePaul': 309, 'Oklahoma St.': 310, 'TCU': 311, "Saint Mary's": 312, 'Rutgers': 313, 'Southern Utah': 314, 'Bryant': 315, 'Vanderbilt': 316, 'San Diego St.': 317, 'Portland St.': 318, 'South Dakota': 319, 'North Carolina A&T': 320, 'Iowa': 321, 'Miami FL': 322, 'Tennessee Tech': 323, 'Santa Clara': 324, 'Cincinnati': 325, 'Georgia Tech': 326, 'Maryland': 327, 'Michigan St.': 328, 'Wisconsin': 329, 'North Alabama': 330, 'Dayton': 331, 'Oregon': 332, 'North Dakota St.': 333, 'SMU': 334, 'Loyola Chicago': 335, 'Valparaiso': 336, 'UC Davis': 337, 'Rhode Island': 338, 'Cal Baptist': 339, 'Howard': 340, 'UCF': 341, 'Duke': 342, 'Gonzaga': 343, 'IU Indy': 344, 'Florida': 345, 'Tennessee': 346, 'Clemson': 347, 'Ohio St.': 348, 'Auburn': 349, 'UT Arlington': 350, 'Syracuse': 351, 'Connecticut': 352, 'Penn St.': 353, 'Butler': 354, 'Purdue': 355, 'George Washington': 356, "Saint Joseph's": 357, 'Nevada': 358, 'Bradley': 359, 'Pittsburgh': 360, 'Alabama': 361, 'Jacksonville St.': 362, 'North Carolina': 363, 'Texas Tech': 364, 'St. Francis NY': 365, 'Hartford': 366, 'Savannah St.': 367, 'Centenary': 368, 'Winston Salem St.': 369}

string_inputs = ['conf', 'team', 'Role', 'Class']

placeholders = {'Height': 78, 'conf': 22, 'Role': 5, 'team': 250, 'Min_per%': 72.9359296482412, 'PRPG!': 3.9653266331658292, 'D-PRPG': 3.798743718592965, 'BPM': 7.209798994974874, 'OBPM': 4.864070351758793, 'DBPM': 2.347361809045226, 'Ortg': 112.71193467336683, 'D-Rtg': 91.86532663316584, 'usg': 24.354020100502513, 'eFG': 54.23052763819096, 'TS_per': 57.8074120603015, 'ORB_per': 6.643341708542714, 'DRB_per': 16.407412060301507, 'AST_per': 15.975125628140704, 'TO_per': 16.376884422110553, 'Blk': 3.285929648241206, 'Stl': 2.112814070351759, 'FTR': 40.39359296482412, 'Dunks%': 0.773825376884422, 'Close 2 %': 0.5839836683417086, 'Far 2 %': 0.33704020100502513, 'FT_per': 0.7387349246231156, '2P%': 0.531178391959799, '3PR': 28.61482412060301, '3P%': 0.3200138190954774, 'Ast': 2.5071608040201006, 'Reb': 6.156532663316582, 'Pts': 15.087311557788945, 'DunksA': 20.045226130653266, 'DunksM': 21.82788944723618, 'Close2A': 75.77386934673366, 'Close2M': 113.83291457286433, 'Far2A': 41.20728643216081, 'Far2M': 106.23618090452261, 'FTA': 105.58040201005025, 'FTM': 142.2424623115578, '2PA': 133.99120603015075, '2PM': 251.6105527638191, '3PA': 40.33040201005025, '3PM': 108.51130653266331}
# Route to handle POST requests and output data
@app.route('/submit', methods=['POST'])
def submit():
    player = "general"
    # Check if data is JSON or form data
    y = request.get_data()
    q = json.loads(y)

    data_dict = q["data"]

    print(data_dict)

    for feature in features:
            if feature not in data_dict:
                data_dict[feature] = placeholders.get(feature, 0) * 0.6

    for key in string_inputs:
        if key == 'conf':
            if data_dict[key] in conf_map:
                data_dict[key] = conf_map[data_dict[key] ]
            else:
                data_dict[key] = len(conf_map)
        elif key == 'team':
            data_dict[key] = team_map[data_dict[key] ]
        elif key == 'Class':
            data_dict[key] = grade_map[data_dict[key] ]
        elif key == 'Role':
            data_dict[key] = position_map[data_dict[key] ]

    dict_df = pd.DataFrame([data_dict]) 

    dict_df = dict_df[features]
    print(dict_df.head())


    predicted_pick = bagging_model.predict(dict_df)[0]
    for col in dict_df.columns:
        dict_df[col] = pd.to_numeric(dict_df[col], errors='coerce')

    will_draft = nn_model.predict(dict_df)[0]
    print(predicted_pick)
    print(player)
    print(will_draft)

    if "playerType" in q:
        player = q["playerType"]


    df = pd.read_csv("cluster_data.csv")
    df = df.sort_values(by='Pick')
    year = 2010
    df = df[df['year'] >= year]
    df = df[df['Pick'] >= 1]

    profile = player
    df = cluster_analysis(df, profile)
    input_stats = []

        
    for stat in profiles[player]["stats"]:
        input_stats.append(float(data_dict[stat]))

    print(input_stats, player)

    membership, predicted_cluster = predict_cluster(player, input_stats)
    print("Predicted Membership Values:", membership)
    print("Predicted Cluster:", predicted_cluster)

    clusters = []
    r = 0
    if player == "slasher":
        r = 3

    elif player == "shooter":
        r = 4

    for i in range(r):
        clusters.append(player + "_membership_" + str(i))

    similarPlayers = df

    predicted_membership = np.array(membership)  # Convert to numpy array

# List to store the distances between the predicted cluster and each player's membership values
    distances = []
    player_stats = ['player_name', 'OBPM', 'DBPM', 'TS_per']

    # Iterate over each player's membership vector in 'similarPlayers'
    for _, row in similarPlayers.iterrows():
        player_membership = row[clusters].values  # Extract the membership values for this player
        distance = np.linalg.norm(predicted_membership - player_membership)  # Euclidean distance
        distances.append((row[player_stats], distance))  # Assuming 'player_name' is the column with player names

    # Sort the players by distance (ascending) and take the top 10 closest
    closest_players = sorted(distances, key=lambda x: x[1])[:10]

    player_comps = []

    player_comps.append({
        "player_name": "Projected Pick",  # Assuming 'player_name' is part of player_stats
        "OBPM": float(data_dict["OBPM"]),
        "DBPM": float(data_dict["DBPM"]),
        "TS_per": float(data_dict["TS_per"]),
        "distance": 0
    })
    for player, distance in closest_players:
        player_comps.append({
            "player_name": player["player_name"],  # Assuming 'player_name' is part of player_stats
            "OBPM": player["OBPM"],
            "DBPM": player["DBPM"],
            "TS_per": player["TS_per"],
            "distance": distance
        })
    
    for player in player_comps:
        print(player)
    
    if will_draft < 0.15:
        predicted_pick = 0

    return jsonify({"predicted_pick": predicted_pick, "player_comps": player_comps, "Status": 200})

if __name__ == "__main__":
    app.run(port=8000, debug=True)