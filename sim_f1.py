import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# Streamlit configuration
st.set_page_config(layout='wide', page_title='F1 Weather Simulator')
st.title('F1 Weather Simulator — simulate race results from weather and historical data')


@st.cache_data
def load_all():
    # load datasets (may raise if missing)
    weather = pd.read_parquet('resources/weather_features_v4 1.parquet')
    races = pd.read_csv('resources/races.csv')
    circuits = pd.read_csv('resources/circuits.csv')
    results = pd.read_csv('resources/results.csv')
    drivers = pd.read_csv('resources/drivers.csv')
    try:
        qualifying = pd.read_csv('resources/qualifying.csv')
    except Exception:
        qualifying = pd.DataFrame()
    # ensure numeric
    results['position'] = pd.to_numeric(results.get('position', pd.Series()), errors='coerce')
    return weather, races, circuits, results, drivers, qualifying


weather, races, circuits, results, drivers, qualifying = load_all()


# ----------------------
# Helper: detect common weather columns
def detect_weather_features(df: pd.DataFrame):
    candidates = {
        'temperature': ['temperature', 'temp', 'air_temp', 't2m'],
        'precipitation': ['precipitation', 'precip', 'rain', 'gfs_precipitations'],
        'windspeed': ['windspeed', 'wind', 'gfs_wind_speed'],
        'round': ['round'],
        'name': ['name'],
        'lat': ['fact_latitude', 'latitude', 'lat'],
        'lon': ['fact_longitude', 'longitude', 'lng', 'lon']
    }
    found = {}
    for k, opts in candidates.items():
        for o in opts:
            if o in df.columns:
                found[k] = o
                break
        else:
            found[k] = None
    return found


wf = detect_weather_features(weather)
weather_features = [wf[f] for f in ['temperature', 'precipitation', 'windspeed'] if wf.get(f)]
if not weather_features:
    st.error('Aucune colonne météo détectée dans le parquet `resources/weather_features_v4 1.parquet`.')
    st.stop()


# Sidebar: global controls
with st.sidebar:
    mode = st.selectbox('Mode (Pilote / Écurie)', ['Pilote', 'Écurie'])
    year = st.selectbox('Année', sorted(races['year'].dropna().unique(), reverse=True))
    available_races = races[races['year'] == year]
    race_choice = st.selectbox('Course', available_races['name'].tolist())
    st.markdown('---')
    st.write('Réglages simulation')
    n_sim = st.slider('Nombre de simulations (Monte‑Carlo)', 50, 2000, 250)
    random_seed = st.number_input('Seed aléatoire (0 = aléatoire)', value=42)


# Prepare historical results for the selected year
races_year = races[races['year'] == year]
results_year = results.merge(races_year[['raceId', 'name', 'round', 'year', 'circuitId']], on='raceId', how='inner')

# Merge weather onto results_year using round+name when available, otherwise nearest circuit coordinates
if wf.get('round') and wf.get('name') and wf['round'] in weather.columns and wf['name'] in weather.columns:
    wk = weather[[wf['round'], wf['name']] + weather_features].groupby([wf['round'], wf['name']]).mean().reset_index()
    results_year = results_year.merge(wk, left_on=['round', 'name'], right_on=[wf['round'], wf['name']], how='left')
else:
    # fallback: map weather points to nearest circuit using lat/lon
    if wf.get('lat') and wf.get('lon') and wf['lat'] in weather.columns and wf['lon'] in weather.columns:
        wc = weather[[wf['lat'], wf['lon']] + weather_features].dropna()
        cc = circuits[['circuitId', 'lat', 'lng']].dropna().reset_index(drop=True)
        if not cc.empty and not wc.empty:
            tree = cKDTree(cc[['lat', 'lng']].values)
            dists, idxs = tree.query(wc[[wf['lat'], wf['lon']]].values, k=1)
            wc = wc.reset_index(drop=True)
            wc['circuitId'] = cc.loc[idxs, 'circuitId'].values
            weather_by_c = wc.groupby('circuitId')[weather_features].mean().reset_index()
            results_year = results_year.merge(weather_by_c, on='circuitId', how='left')


# Parse qualifying times to seconds and merge best lap
if not qualifying.empty:
    def parse_time_to_seconds(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == '' or s == '\\N':
            return np.nan
        try:
            if ':' in s:
                minutes, sec = s.split(':', 1)
                return float(minutes) * 60.0 + float(sec)
            else:
                return float(s)
        except Exception:
            return np.nan

    for c in ['q1', 'q2', 'q3']:
        if c in qualifying.columns:
            qualifying[c + '_sec'] = qualifying[c].apply(parse_time_to_seconds)
        else:
            qualifying[c + '_sec'] = np.nan

    qualifying['best_q'] = qualifying[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)
    qual_best = qualifying.groupby(['raceId', 'driverId'])['best_q'].min().reset_index()
    results_year = results_year.merge(qual_best, on=['raceId', 'driverId'], how='left')


# Prepare driver-level stats (overall and wet)
prec_col = wf.get('precipitation')
if prec_col and prec_col in results_year.columns:
    results_year['_precip'] = pd.to_numeric(results_year[prec_col], errors='coerce').fillna(0.0)
else:
    results_year['_precip'] = 0.0

results_year['position'] = pd.to_numeric(results_year.get('position', pd.Series()), errors='coerce')

general = results_year.groupby('driverId')['position'].mean().reset_index().rename(columns={'position': 'avg_pos_all'})
wet = results_year[results_year['_precip'] > 0].groupby('driverId')['position'].mean().reset_index().rename(columns={'position': 'avg_pos_wet'})
driver_stats = general.merge(wet, on='driverId', how='left')
driver_stats['avg_pos_wet'] = driver_stats['avg_pos_wet'].fillna(driver_stats['avg_pos_all'])


# Main UI: selection panel and weather overrides
col_main, col_side = st.columns([3, 1])
with col_side:
    st.header('Résumé')
    st.write('Année:', year)
    st.write('Course:', race_choice)
    st.write('Mode:', mode)
    st.write('Données météo détectées:', ', '.join(weather_features))

with col_main:
    st.header('Sélection Pilotes / Écuries')
    if mode == 'Pilote':
        # Build a table of drivers who actually raced in the selected year
        res_drivers = results_year[['driverId', 'position']].dropna()
        if not res_drivers.empty:
            driver_counts = res_drivers.groupby('driverId').agg(races_count=('driverId','count'), avg_pos=('position','mean')).reset_index()
        else:
            driver_counts = pd.DataFrame(columns=['driverId','races_count','avg_pos'])

        # Merge with drivers master table to get names
        driver_master = drivers.copy()
        driver_master['label'] = driver_master['forename'].fillna('') + ' ' + driver_master['surname'].fillna('')
        # ensure driverId types align
        try:
            driver_master['driverId'] = driver_master['driverId'].astype(int)
        except Exception:
            driver_master['driverId'] = driver_master['driverId']

        drv_tab = driver_master.merge(driver_counts, on='driverId', how='left').fillna({'races_count':0, 'avg_pos':np.nan})
        drv_tab_display = drv_tab[['driverId','label','races_count','avg_pos']].sort_values(['races_count','avg_pos'], ascending=[False, True])

        st.write('Pilotes ayant couru cette année (tableau) — cochez dans la liste ci-dessous pour sélectionner:')
        st.dataframe(drv_tab_display.reset_index(drop=True))

        # Pre-fill multiselect with drivers who have at least 1 race that year
        available_driver_labels = drv_tab_display[drv_tab_display['races_count']>0]['label'].tolist()
        default_selection = available_driver_labels[:10]
        selected = st.multiselect('Sélectionne pilotes', available_driver_labels, default=default_selection)
    else:
        constructors = pd.read_csv('resources/constructors.csv')
        constructors['name'] = constructors['name'].fillna('')
        selected = st.multiselect('Sélectionne écuries', constructors['name'].tolist(), default=constructors['name'].tolist()[:6])

    st.markdown('---')
    st.subheader('Modifier les conditions météo (simulateur)')
    weather_inputs = {}
    for col in weather_features:
        label = col.replace('_', ' ').capitalize()
        if 'temp' in col:
            default = float(weather[col].dropna().mean()) if col in weather.columns and not weather[col].dropna().empty else 20.0
            weather_inputs[col] = st.slider(f'{label} (°C)', -20.0, 50.0, default)
        elif 'precip' in col or 'rain' in col:
            default = float(weather[col].dropna().mean()) if col in weather.columns and not weather[col].dropna().empty else 0.0
            weather_inputs[col] = st.slider(f'{label} (mm)', 0.0, 300.0, default)
        elif 'wind' in col:
            default = float(weather[col].dropna().mean()) if col in weather.columns and not weather[col].dropna().empty else 10.0
            weather_inputs[col] = st.slider(f'{label} (km/h)', 0.0, 200.0, default)
        else:
            default = float(weather[col].dropna().mean()) if col in weather.columns and not weather[col].dropna().empty else 0.0
            weather_inputs[col] = st.number_input(label, value=default)


# Build training set for model (features present in results_year)
feat_cols = [c for c in weather_features if c in results_year.columns]
train = results_year.dropna(subset=['position']).dropna(subset=feat_cols, how='all')
if train.empty:
    st.warning('Pas assez de données historiques pour entraîner un modèle sur cette année. Les résultats seront approximatifs.')

# merge driver stats into train
train = train.merge(driver_stats, on='driverId', how='left')
# fill missing
if 'avg_pos_all' in train.columns:
    train['avg_pos_all'] = train['avg_pos_all'].fillna(train['avg_pos_all'].mean())
if 'avg_pos_wet' in train.columns:
    train['avg_pos_wet'] = train['avg_pos_wet'].fillna(train['avg_pos_wet'].mean())

# encode driverId for model
le = LabelEncoder()
try:
    train['driver_enc'] = le.fit_transform(train['driverId'].astype(str))
except Exception:
    train['driver_enc'] = 0

# define model features
model_features = feat_cols + ['grid', 'best_q', 'avg_pos_all', 'avg_pos_wet', 'driver_enc']
model_features = [f for f in model_features if f in train.columns]

# train a simple regressor mapping features -> finishing position
if not train.empty and not set(model_features).intersection(train.columns):
    st.warning('Aucune caractéristique utilisable pour l\'entraînement du modèle.')

if not train.empty:
    X = train[model_features]
    y = train['position']
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)
else:
    model = None


# prepare lineup based on selection
lineup = []
if mode == 'Pilote':
    driver_df = drivers.copy()
    driver_df['label'] = driver_df['forename'].fillna('') + ' ' + driver_df['surname'].fillna('')
    driver_df['driverId'] = driver_df['driverId'].astype(int)
    for lbl in selected:
        row = driver_df[driver_df['label'] == lbl]
        if not row.empty:
            did = int(row['driverId'].iloc[0])
            lineup.append({'label': lbl, 'driverId': did, 'grid': 10})
else:
    # for constructors: expand into drivers who raced for them in the selected year
    res_year = results_year[results_year['year'] == year]
    cons = pd.read_csv('resources/constructors.csv')
    name_to_id = dict(zip(cons['name'], cons['constructorId']))
    chosen = [name_to_id.get(n) for n in selected]
    chosen = [c for c in chosen if c is not None]
    drows = res_year[res_year['constructorId'].isin(chosen)][['driverId']].drop_duplicates().head(30)
    for _, r in drows.iterrows():
        did = int(r['driverId'])
        dr = drivers[drivers['driverId'] == did]
        if not dr.empty:
            lbl = dr['forename'].iloc[0] + ' ' + dr['surname'].iloc[0]
            lineup.append({'label': lbl, 'driverId': did, 'grid': 10})


st.subheader('Grille (optionnel)')
for p in lineup:
    p['grid'] = st.number_input(f"Position grille {p['label']}", min_value=1, max_value=30, value=p['grid'])


# Simulation logic: Monte-Carlo over n_sim runs, aggregating finishing positions
def simulate_once(model, model_features, feat_cols, weather_inputs, driver_stats, train, lineup):
    rows = []
    for p in lineup:
        d = p['driverId']
        row = {}
        for c in feat_cols:
            row[c] = float(weather_inputs.get(c, 0.0))
        row['grid'] = float(p.get('grid', 10))
        # best_q: use driver's mean best_q in train if available
        if 'best_q' in train.columns:
            qq = train[train['driverId'] == d]['best_q']
            row['best_q'] = float(qq.mean()) if not qq.empty else float(train['best_q'].mean())
        if isinstance(driver_stats, pd.DataFrame):
            stats_row = driver_stats[driver_stats['driverId'] == d]
            if not stats_row.empty:
                row['avg_pos_all'] = float(stats_row['avg_pos_all'].iloc[0])
                row['avg_pos_wet'] = float(stats_row['avg_pos_wet'].iloc[0])
            else:
                row['avg_pos_all'] = float(train['avg_pos_all'].mean()) if 'avg_pos_all' in train.columns else 15.0
                row['avg_pos_wet'] = float(train['avg_pos_wet'].mean()) if 'avg_pos_wet' in train.columns else 15.0
        else:
            row['avg_pos_all'] = 15.0
            row['avg_pos_wet'] = 15.0
        try:
            row['driver_enc'] = int(le.transform([str(d)])[0])
        except Exception:
            row['driver_enc'] = 0
        rows.append((p['label'], d, row))

    dfp = pd.DataFrame([r[2] for r in rows])
    # Keep only features used by model
    Xp = dfp[[f for f in model_features if f in dfp.columns]]
    # model predicts expected finishing position; add random noise to simulate variability
    preds = model.predict(Xp) if model is not None else np.random.normal(15, 5, size=len(Xp))
    # add small gaussian noise
    preds = preds + np.random.normal(0, 1.5, size=len(preds))
    # return list of tuples (label, predicted position)
    return [(rows[i][0], float(preds[i])) for i in range(len(rows))]


if st.button('Simuler la course'):
    if not lineup:
        st.warning('Sélectionne au moins un pilote ou une écurie.')
    else:
        # set RNG
        if random_seed != 0:
            np.random.seed(int(random_seed))

        # run Monte-Carlo
        accum = {p['label']: [] for p in lineup}
        for i in range(int(n_sim)):
            out = simulate_once(model, model_features, feat_cols, weather_inputs, driver_stats, train, lineup)
            # sort by predicted position ascending -> best first
            ordered = sorted(out, key=lambda x: x[1])
            for pos, (label, _) in enumerate(ordered, start=1):
                accum[label].append(pos)

        # compute statistics: mean finishing position and podium probability
        stats = []
        for label, positions in accum.items():
            mean_pos = float(np.mean(positions))
            podium_prob = float(np.mean([1 if p <= 3 else 0 for p in positions]))
            win_prob = float(np.mean([1 if p == 1 else 0 for p in positions]))
            stats.append({'label': label, 'mean_pos': mean_pos, 'podium_prob': podium_prob, 'win_prob': win_prob})

        df_stats = pd.DataFrame(stats).sort_values('mean_pos')

        st.subheader('Classement simulé (moyenne des positions)')
        st.table(df_stats[['label', 'mean_pos', 'win_prob', 'podium_prob']].rename(columns={'label': 'Pilote', 'mean_pos': 'Position moyenne', 'win_prob': "Prob victoire", 'podium_prob': "Prob podium"}))

        st.subheader('Distribution des positions (exemples)')
        # show histogram for top 5
        top5 = df_stats.head(5)['label'].tolist()
        for t in top5:
            st.write(f"{t} — position moyenne: {df_stats[df_stats['label']==t]['mean_pos'].iloc[0]:.2f}")

