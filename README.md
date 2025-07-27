# Phase1-Project-CG

Load the Dataset with Pandas That is import:

pandas with the standard alias pd
matplotlib.pyplot with the standard alias plt And set %matplotlib inline so the graphs will display immediately below the cell that creates them.

```python
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

Use pandas to open the csv file and name the resulting dataframe df and check that you loaded the data correctly

```python
df = pd.read_csv("/content/Aviation_Data.csv", low_memory=False)
df
```
Data exploration:
Inspect the contents of the dataframe

```python
df.shape
(9194, 31)
```
```python
df.columns
```python
Index(['Event.Id', 'Investigation.Type', 'Accident.Number', 'Event.Date',
       'Location', 'Country', 'Latitude', 'Longitude', 'Airport.Code',
       'Airport.Name', 'Injury.Severity', 'Aircraft.damage',
       'Aircraft.Category', 'Registration.Number', 'Make', 'Model',
       'Amateur.Built', 'Number.of.Engines', 'Engine.Type', 'FAR.Description',
       'Schedule', 'Purpose.of.flight', 'Air.carrier', 'Total.Fatal.Injuries',
       'Total.Serious.Injuries', 'Total.Minor.Injuries', 'Total.Uninjured',
       'Weather.Condition', 'Broad.phase.of.flight', 'Report.Status',
       'Publication.Date'],
      dtype='object')

View the first five records
```python
df.head(5)
```

View dataset information
```python
df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9194 entries, 0 to 9193
Data columns (total 31 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   Event.Id                9194 non-null   object 
 1   Investigation.Type      9194 non-null   object 
 2   Accident.Number         9194 non-null   object 
 3   Event.Date              9194 non-null   object 
 4   Location                9191 non-null   object 
 5   Country                 9155 non-null   object 
 6   Latitude                5 non-null      float64
 7   Longitude               5 non-null      float64
 8   Airport.Code            4640 non-null   object 
 9   Airport.Name            5556 non-null   object 
 10  Injury.Severity         9194 non-null   object 
 11  Aircraft.damage         9014 non-null   object 
 12  Aircraft.Category       3581 non-null   object 
 13  Registration.Number     9193 non-null   object 
 14  Make                    9187 non-null   object 
 15  Model                   9179 non-null   object 
 16  Amateur.Built           9192 non-null   object 
 17  Number.of.Engines       9100 non-null   float64
 18  Engine.Type             9191 non-null   object 
 19  FAR.Description         3581 non-null   object 
 20  Schedule                1604 non-null   object 
 21  Purpose.of.flight       9182 non-null   object 
 22  Air.carrier             418 non-null    object 
 23  Total.Fatal.Injuries    9149 non-null   float64
 24  Total.Serious.Injuries  9138 non-null   float64
 25  Total.Minor.Injuries    9134 non-null   float64
 26  Total.Uninjured         9170 non-null   float64
 27  Weather.Condition       9192 non-null   object 
 28  Broad.phase.of.flight   9183 non-null   object 
 29  Report.Status           9193 non-null   object 
 30  Publication.Date        3638 non-null   object 
dtypes: float64(7), object(24)
memory usage: 2.2+ MB

Data Cleaning & Missing Value Imputation. This entails:
Dropping unnecessary columns or rows
Filling missing values (with mean, median, mode, or a constant)
Standardizing formats (e.g., date formats, capitalization)
Removing duplicates
Handling outliers

```python
# check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print("Missing values:\n", missing)
Missing values:
 Latitude                  9189
Longitude                 9189
Air.carrier               8776
Schedule                  7590
FAR.Description           5613
Aircraft.Category         5613
Publication.Date          5556
Airport.Code              4554
Airport.Name              3638
Aircraft.damage            180
Number.of.Engines           94
Total.Minor.Injuries        60
Total.Serious.Injuries      56
Total.Fatal.Injuries        45
Country                     39
Total.Uninjured             24
Model                       15
Purpose.of.flight           12
Broad.phase.of.flight       11
Make                         7
Engine.Type                  3
Location                     3
Amateur.Built                2
Weather.Condition            2
Registration.Number          1
Report.Status                1
Injury.Severity              0
Accident.Number              0
Investigation.Type           0
Event.Id                     0
Event.Date                   0
dtype: int64
```
```python
# Drop irrelevant or consistently missing columns (example)
df.drop(columns=['investigation_type', 'publication_date'], inplace=True, errors='ignore')
df
```
```python
df.shape
(9194, 31)
```

```python
# Check percentage of missing values in latitude and longitude
missing_geo = df[['Latitude', 'Longitude']].isna().mean() * 100
print(missing_geo)
Latitude     99.945617
Longitude    99.945617
dtype: float64
```
```python
df.drop(columns=['airport_code', 'airport_name','aircraft_category', 'air_carrier','latitude','longitude'], inplace=True, errors='ignore')
df
```
Standardize or Normalize all Column names

```python
df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9194 entries, 0 to 9193
Data columns (total 31 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   Event.Id                9194 non-null   object 
 1   Investigation.Type      9194 non-null   object 
 2   Accident.Number         9194 non-null   object 
 3   Event.Date              9194 non-null   object 
 4   Location                9191 non-null   object 
 5   Country                 9155 non-null   object 
 6   Latitude                5 non-null      float64
 7   Longitude               5 non-null      float64
 8   Airport.Code            4640 non-null   object 
 9   Airport.Name            5556 non-null   object 
 10  Injury.Severity         9194 non-null   object 
 11  Aircraft.damage         9014 non-null   object 
 12  Aircraft.Category       3581 non-null   object 
 13  Registration.Number     9193 non-null   object 
 14  Make                    9187 non-null   object 
 15  Model                   9179 non-null   object 
 16  Amateur.Built           9192 non-null   object 
 17  Number.of.Engines       9100 non-null   float64
 18  Engine.Type             9191 non-null   object 
 19  FAR.Description         3581 non-null   object 
 20  Schedule                1604 non-null   object 
 21  Purpose.of.flight       9182 non-null   object 
 22  Air.carrier             418 non-null    object 
 23  Total.Fatal.Injuries    9149 non-null   float64
 24  Total.Serious.Injuries  9138 non-null   float64
 25  Total.Minor.Injuries    9134 non-null   float64
 26  Total.Uninjured         9170 non-null   float64
 27  Weather.Condition       9192 non-null   object 
 28  Broad.phase.of.flight   9183 non-null   object 
 29  Report.Status           9193 non-null   object 
 30  Publication.Date        3638 non-null   object 
dtypes: float64(7), object(24)
memory usage: 2.2+ MB

```python
#Clean Column Names
# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace('.', '_').str.replace(' ', '_')

# Check again
print(df.columns.tolist())
['event_id', 'investigation_type', 'accident_number', 'event_date', 'location', 'country', 'latitude', 'longitude', 'airport_code', 'airport_name', 'injury_severity', 'aircraft_damage', 'aircraft_category', 'registration_number', 'make', 'model', 'amateur_built', 'number_of_engines', 'engine_type', 'far_description', 'schedule', 'purpose_of_flight', 'air_carrier', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries', 'total_uninjured', 'weather_condition', 'broad_phase_of_flight', 'report_status', 'publication_date']

```
```python
df['event_date']
```

```python
# Convert dates
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df[df['event_date'].notnull()]  # remove rows with invalid dates
```

```python
df.head(3)
```

```python
df['aircraft_damage']
```

```python
df['engine_type']
```

```python
df['make']
```

```python
# Fill missing values
df['aircraft_damage'].fillna('Unknown', inplace=True)
df['engine_type'].fillna('Unknown', inplace=True)
df['make'].fillna('Unknown', inplace=True)
```
```python
df.loc[:, 'aircraft_damage'] = df['aircraft_damage'].fillna('Unknown')
df.loc[:, 'engine_type'] = df['engine_type'].fillna('Unknown')
df.loc[:, 'make'] = df['make'].fillna('Unknown')
```
Exploratory Data Analysis (EDA) Analyze aircraft type, fatal vs. non-fatal accidents, engine type, and manufacturer risk.

Check the actual column names

```python
print(df.columns.tolist())
['event_id', 'investigation_type', 'accident_number', 'event_date', 'location', 'country', 'latitude', 'longitude', 'airport_code', 'airport_name', 'injury_severity', 'aircraft_damage', 'aircraft_category', 'registration_number', 'make', 'model', 'amateur_built', 'number_of_engines', 'engine_type', 'far_description', 'schedule', 'purpose_of_flight', 'air_carrier', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries', 'total_uninjured', 'weather_condition', 'broad_phase_of_flight', 'report_status', 'publication_date']
Risk Aggregation - Define risk based on fatalities, injuries, damage severity etc
```

```python
# Make a safe copy of your DataFrame (important if df was filtered from another one)
df = df.copy()

# Fill missing values
df['aircraft_damage'].fillna('Unknown', inplace=True)
df['engine_type'].fillna('Unknown', inplace=True)
df['make'].fillna('Unknown', inplace=True)

# Create a risk score: fatalities + serious injuries weighted more
df['risk_score'] = (
    df['total_fatal_injuries'].fillna(0)*3 +
    df['total_serious_injuries'].fillna(0)*2 +
    df['total_minor_injuries'].fillna(0)*1
)

# Group by aircraft make and calculate average risk score and total incidents
risk_summary = df.groupby('make').agg({
    'risk_score': 'mean',
    'event_id': 'count'
}).rename(columns={'event_id': 'total_incidents'}).sort_values(by='risk_score')

# Display the 10 makes with the lowest average risk score
print(risk_summary.head(10))


                 risk_score  total_incidents
make                                        
Young-losey             0.0                1
Auster                  0.0                1
Viking                  0.0                1
Vultee                  0.0                1
Wainscott               0.0                1
Piper-aerostar          0.0                1
Pilatus                 0.0                1
Pietenpol-grega         0.0                1
Aerostar                0.0                1
Pratt-read              0.0                1
/tmp/ipython-input-23-480803365.py:5: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['aircraft_damage'].fillna('Unknown', inplace=True)
/tmp/ipython-input-23-480803365.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['engine_type'].fillna('Unknown', inplace=True)
/tmp/ipython-input-23-480803365.py:7: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['make'].fillna('Unknown', inplace=True)
```
```python
# Aircraft accident count by manufacturer
top_makes = df['make'].value_counts().head(10)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x=top_makes.values, y=top_makes.index)
plt.title("Top 10 Aircraft Manufacturers by Number of Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Manufacturer")
plt.tight_layout()
plt.show()

to check aircraft make, model, and their corresponding risk score and injury counts.
```

```python
# Filter valid dates and create a safe copy
df = df[df['event_date'].notnull()].copy()
```
```python
# Fill missing injury values
df.loc[:, 'total_fatal_injuries'] = df['total_fatal_injuries'].fillna(0)
df.loc[:, 'total_serious_injuries'] = df['total_serious_injuries'].fillna(0)
df.loc[:, 'total_minor_injuries'] = df['total_minor_injuries'].fillna(0)

# Calculate risk score
df.loc[:, 'risk_score'] = (
    df['total_fatal_injuries'] * 3 +
    df['total_serious_injuries'] * 2 +
    df['total_minor_injuries'] * 1
)

# Confirm changes by printing first few rows
print(df[['make', 'model', 'risk_score', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries']].head())

       make     model  risk_score  total_fatal_injuries  \
0   Stinson     108-3         6.0                   2.0   
1     Piper  PA24-180        12.0                   4.0   
2    Cessna      172M         9.0                   3.0   
3  Rockwell       112         6.0                   2.0   
4    Cessna       501         7.0                   1.0   

   total_serious_injuries  total_minor_injuries  
0                     0.0                   0.0  
1                     0.0                   0.0  
2                     0.0                   0.0  
3                     0.0                   0.0  
4                     2.0                   0.0  
See Top Low-Risk Manufacturers (Summary Table)
```

```python
# Top low-risk Manufacures(Summary Table)
# Group by aircraft manufacturer
risk_summary = df.groupby('make').agg({
    'risk_score': 'mean',
    'event_id': 'count'
}).rename(columns={'event_id': 'total_incidents'}).sort_values(by='risk_score')

# Show top 10 manufacturers with lowest average risk
print(risk_summary.head(10))

                 risk_score  total_incidents
make                                        
Young-losey             0.0                1
Auster                  0.0                1
Viking                  0.0                1
Vultee                  0.0                1
Wainscott               0.0                1
Piper-aerostar          0.0                1
Pilatus                 0.0                1
Pietenpol-grega         0.0                1
Aerostar                0.0                1
Pratt-read              0.0                1
```
```python
print("SHAPE OF DF:", df.shape)
print(df[['make', 'risk_score']].dropna().head(10))

SHAPE OF DF: (9194, 32)
                make  risk_score
0            Stinson         6.0
1              Piper        12.0
2             Cessna         9.0
3           Rockwell         6.0
4             Cessna         7.0
5  Mcdonnell Douglas         1.0
6             Cessna        12.0
7             Cessna         0.0
8             Cessna         0.0
9     North American         3.0
Top 10 Lowest Risk Manufacturers - Visualization. Lowest Rist Manufactures
```

```python
# Make a safe copy of your DataFrame (important if df was filtered from another one)
df = df.copy()
```
# Fill missing values
df['aircraft_damage'].fillna('Unknown', inplace=True)
df['engine_type'].fillna('Unknown', inplace=True)
df['make'].fillna('Unknown', inplace=True)

# Create a risk score: fatalities + serious injuries weighted more
df['risk_score'] = (
    df['total_fatal_injuries'].fillna(0)*3 +
    df['total_serious_injuries'].fillna(0)*2 +
    df['total_minor_injuries'].fillna(0)*1
)

# Group by aircraft make and calculate average risk score and total incidents
risk_summary = df.groupby('make').agg({
    'risk_score': 'mean',
    'event_id': 'count'
}).rename(columns={'event_id': 'total_incidents'}).sort_values(by='risk_score')

# Display the 10 makes with the lowest average risk score
print(risk_summary.head(10))


                 risk_score  total_incidents
make                                        
Young-losey             0.0                1
Auster                  0.0                1
Viking                  0.0                1
Vultee                  0.0                1
Wainscott               0.0                1
Piper-aerostar          0.0                1
Pilatus                 0.0                1
Pietenpol-grega         0.0                1
Aerostar                0.0                1
Pratt-read              0.0                1
/tmp/ipython-input-29-480803365.py:5: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['aircraft_damage'].fillna('Unknown', inplace=True)
/tmp/ipython-input-29-480803365.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['engine_type'].fillna('Unknown', inplace=True)
/tmp/ipython-input-29-480803365.py:7: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['make'].fillna('Unknown', inplace=True)
Incidents vs. Average Risk score

```
```python
import plotly.express as px

# Prepare summary DataFrame for scatter
risk_plot_df = risk_summary.reset_index()

fig = px.scatter(
    risk_plot_df,
    x='total_incidents',
    y='risk_score',
    hover_name='make',
    size='total_incidents',
    color='risk_score',
    color_continuous_scale='RdYlGn_r',
    title='Aircraft Manufacturers: Incidents vs. Risk Score'
)
fig.update_layout(xaxis_title='Total Incidents', yaxis_title='Average Risk Score')
fig.show()

```
Objective to find our the airline with less accident Aircfraft by model, category, flight

which aircraft types are involved in the most accidents

```python
accidents_by_type = df['model'].value_counts()
print(accidents_by_type.head(10))
model
152          381
172          206
150          151
172N         137
PA-28-140    135
PA-38-112    102
172M          92
150M          89
G-164A        89
182           82
Name: count, dtype: int64
Visualization
```
```python
import matplotlib.pyplot as plt

top_models = accidents_by_type.head(10)

plt.figure(figsize=(10, 6))
top_models.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Accidents')
plt.ylabel('Aircraft Model')
plt.title('Top 10 Aircraft Models by Accident Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

```

```python
print(df['investigation_type'].unique())

['Accident' 'Incident']
```
```python
print(df.columns.tolist())
['event_id', 'investigation_type', 'accident_number', 'event_date', 'location', 'country', 'latitude', 'longitude', 'airport_code', 'airport_name', 'injury_severity', 'aircraft_damage', 'aircraft_category', 'registration_number', 'make', 'model', 'amateur_built', 'number_of_engines', 'engine_type', 'far_description', 'schedule', 'purpose_of_flight', 'air_carrier', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries', 'total_uninjured', 'weather_condition', 'broad_phase_of_flight', 'report_status', 'publication_date', 'risk_score']
```
Filter only accidents (exclude incidents)
```python
accidents_df = df[df['investigation_type'] == 'Accident']  # ✅ correct
accidents_df
```
```python
print(accidents_df['investigation_type'].value_counts())
investigation_type
Accident    8857
Name: count, dtype: int64
```
```python
# Step 5: Count number of accident records
num_accidents = accidents_df.shape[0]

print(f"Total number of accidents: {num_accidents}")
Total number of accidents: 8857
```
```python
# Correct column names
accidents_by_make_model = accidents_df.groupby(['make', 'model']).size().reset_index(name='accident_count')
accidents_by_make_model
```

```python
# Sort by number of accidents in descending order
accidents_by_make_model_sorted = accidents_by_make_model.sort_values(by='accident_count', ascending=False)
accidents_by_make_model_sorted
```

```python
# By manufacturer (Make)
df['make'].value_counts()
```

```python
# Or by specific aircraft model
df['model'].value_counts()
```

```python
accidents_df = df[df['investigation_type'] == 'Accident']
accidents_df['make'].value_counts()
```

```python
# Display the top 10 combinations
accidents_by_make_model_sorted.head
```

```python
print(df.columns.tolist())
['event_id', 'investigation_type', 'accident_number', 'event_date', 'location', 'country', 'latitude', 'longitude', 'airport_code', 'airport_name', 'injury_severity', 'aircraft_damage', 'aircraft_category', 'registration_number', 'make', 'model', 'amateur_built', 'number_of_engines', 'engine_type', 'far_description', 'schedule', 'purpose_of_flight', 'air_carrier', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries', 'total_uninjured', 'weather_condition', 'broad_phase_of_flight', 'report_status', 'publication_date', 'risk_score']
```
```python
# Prepare data safely
top10 = accidents_by_make_model_sorted.head(10).copy()
bottom10 = accidents_by_make_model_sorted[accidents_by_make_model_sorted['accident_count'] > 0].tail(10).copy()

# Create aircraft label
top10['aircraft'] = top10['make'] + " " + top10['model']
bottom10['aircraft'] = bottom10['make'] + " " + bottom10['model']

# Set up subplots
fig, axes = plt.subplots(ncols=2, figsize=(18, 8))

# Top 10 plot
sns.barplot(data=top10, x='accident_count', y='aircraft', ax=axes[0])
axes[0].set_title("Top 10 Aircraft Make-Models by Accident Count")
axes[0].set_xlabel("accident_count")
axes[0].set_ylabel("")

# Bottom 10 plot
sns.barplot(data=bottom10, x='accident_count', y='aircraft', ax=axes[1])
axes[1].set_title("Bottom 10 Aircraft Make-Models by Accident Count")
axes[1].set_xlabel("accident_count")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Group by make & model, count number of events
accidents_by_make_model_sorted = (
    df.groupby(['make', 'model'])
    .agg({'event_id': 'count'})
    .reset_index()
    .rename(columns={'event_id': 'Accident Count'})
    .sort_values(by='Accident Count', ascending=False)
)

# Step 2: Prepare top and bottom 10 safely
top10 = accidents_by_make_model_sorted.head(10).copy()
bottom10 = accidents_by_make_model_sorted[accidents_by_make_model_sorted['Accident Count'] > 0].tail(10).copy()

# Step 3: Create aircraft label (Make + Model)
top10['Aircraft'] = top10['make'] + " " + top10['model']
bottom10['Aircraft'] = bottom10['make'] + " " + bottom10['model']

# Step 4: Plotting
fig, axes = plt.subplots(ncols=2, figsize=(18, 8))

# Left: Top 10
sns.barplot(data=top10, x='Accident Count', y='Aircraft', ax=axes[0], palette="Blues_d")
axes[0].set_title("Top 10 Aircraft Make-Models by Accident Count")
axes[0].set_xlabel("Accident Count")
axes[0].set_ylabel("Aircraft")

# Right: Bottom 10
sns.barplot(data=bottom10, x='Accident Count', y='Aircraft', ax=axes[1], palette="Greens_d")
axes[1].set_title("Bottom 10 Aircraft Make-Models by Accident Count")
axes[1].set_xlabel("Accident Count")
axes[1].set_ylabel("Aircraft")

plt.tight_layout()
plt.show()

```
```python
#common causes of accidents
print(df.columns.tolist())
['event_id', 'investigation_type', 'accident_number', 'event_date', 'location', 'country', 'latitude', 'longitude', 'airport_code', 'airport_name', 'injury_severity', 'aircraft_damage', 'aircraft_category', 'registration_number', 'make', 'model', 'amateur_built', 'number_of_engines', 'engine_type', 'far_description', 'schedule', 'purpose_of_flight', 'air_carrier', 'total_fatal_injuries', 'total_serious_injuries', 'total_minor_injuries', 'total_uninjured', 'weather_condition', 'broad_phase_of_flight', 'report_status', 'publication_date', 'risk_score']
```

```python
# Check for common phases of flight where accidents occurred
top_causes = df['broad_phase_of_flight'].value_counts().head(10)

# Display the results
print("Top 10 Common Phases of Flight in Accidents:")
print(top_causes)

Top 10 Common Phases of Flight in Accidents:
broad_phase_of_flight
Landing        2365
Takeoff        1923
Cruise         1525
Maneuvering    1189
Approach        980
Taxi            311
Climb           238
Descent         208
Go-around       172
Unknown         135
Name: count, dtype: int64
```
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Filter only Accident cases
accidents_df = df[df["investigation_type"] == "Accident"]

# Drop missing values for meaningful counts
phase_counts = accidents_df["broad_phase_of_flight"].dropna().value_counts()
weather_counts = accidents_df["weather_condition"].dropna().value_counts()

# Plot 1: Broad phase of flight
plt.figure(figsize=(10,6))
sns.barplot(x=phase_counts.values, y=phase_counts.index, palette="coolwarm")
plt.title("Most Common Flight Phases During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Flight Phase")
plt.tight_layout()
plt.show()

# Plot 2: Weather conditions
plt.figure(figsize=(6,4))
sns.barplot(x=weather_counts.index, y=weather_counts.values, palette="muted")
plt.title("Accidents by Weather Condition")
plt.xlabel("weather_condition")
plt.ylabel("accident_number")
plt.tight_layout()
plt.show()

Saving the data to a clean version

```
```python
df.to_csv('Cleaned_Aviation_Datacg.csv', index=False)
```
```python
from google.colab import files
files.download('Cleaned_Aviation_Datacg.csv')
```






