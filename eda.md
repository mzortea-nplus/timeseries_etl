```python
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import duckdb
import yaml
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import matplotlib.cm as cm
import matplotlib.colors as mcolors
cmap = cm.get_cmap("viridis", 6)
norm = mcolors.Normalize(vmin=0, vmax=6 - 1)

```

    C:\Users\m.zortea\AppData\Local\Temp\ipykernel_35668\1770879672.py:9: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
      cmap = cm.get_cmap("viridis", 6)
    

# Data Loading and Preparation

Carico dati dal database utilizzando il dataset di Sommacampagna


```python
with open('configs/SINDY.yaml', "r") as f:
    config = yaml.safe_load(f)
conn = duckdb.connect(database=config["data"]["db"], read_only=True)
_df = conn.execute(
    f"""
    SELECT * FROM main_staging.all_static
    WHERE time between '{config['data']['start_date']}' and '{config['data']['end_date']}'
"""
).df()

    
```

Controllo se ci sono buchi nei dati (timestamp mancanti)


```python
# fix irregular sampling
dts = _df["time"].diff().dt.total_seconds()
dt = dts.mode().iloc[0]
mask = (dts != dt)
if any(mask[1:]):
    print("Found irregular sampling")
    print(_df["time"][mask])
# make a copy
df = _df.copy()

# ensure datetime index
df = df.set_index("time")
df.index = pd.to_datetime(df.index)

# resample / align to 15-minute regular grid
df = df.asfreq('15min')              # introduces NaNs at missing timestamps
df = df.interpolate(method='time')   # fills NaNs using time-based interpolation

df = df.reset_index()

```

    Found irregular sampling
    0      2025-10-01
    4416   2025-11-18
    Name: time, dtype: datetime64[us]
    

Separo il dataset dei termometri dagli altri sensori


```python
tmp_sensors = [s for s in df.columns if s.endswith("_t")]
sensors = [c for c in df.columns if c not in ["dt", "time", "month"] + tmp_sensors]
df_tmp = df[tmp_sensors + ['time']].copy().reset_index()
df_data = df[sensors + ['time']].copy().reset_index()
```

Definisco il numero di punti/misure necessari per coprire gli intervalli desiderati


```python
dt = dts.mode().iloc[0]  # seconds per sample
season_seconds = 24 * 3600  # example: daily seasonality
period = int(round(season_seconds / dt))
```

# Seasonal-Trend Decomposition

Esempio semplice di decomposizione stagionale utilizzando semplici medie mobili per il calcolo del trend.  
Confrontiamo due modelli:
- **additivo** data = trend + seasonal + noise  
- **moltiplicativo** data = trend * seasonal + noise  

Visualizziamo innanzitutto la differenza tra i due casi con un toy model


```python
x = np.linspace(0, 20, 1000)
trend = 2*x - 1
seasonal = 5*np.sin(2*x)
noise = np.random.normal(loc=0, scale=0.2, size=(1000,))

additive = trend + seasonal + noise
multiplicative = trend * seasonal + noise

plt.plot(x, additive, label='additive')
plt.plot(x, multiplicative, label='multiplicative')
plt.grid()
plt.xlabel('x')
plt.legend()
plt.tight_layout()
plt.show()

```


    
![png](eda_files/eda_12_0.png)
    


Nel modello additivo, il trend può essere visto come la media locale della serie temporale e le fluttuazioni come variazioni attorno alla media. Nel modello moltiplicativo, invece, la media rimane sempre nulla e il trend determina solamente l'ampiezza delle fluttuazioni.  
  
Analizziamo il caso di trend e stagionalità entrambi periodici, come può essere ad esempio la concorrenza di due eventi periodici: uno di periodo giornaliero e l'altro annuale. 


```python
x = np.linspace(0, 20, 1000)
trend = np.sin(0.5*x)
seasonal = 0.2*np.sin(20*x)
noise = np.random.normal(loc=0, scale=0.05, size=(1000,))

additive = trend + seasonal + noise
multiplicative = trend * seasonal + noise

plt.plot(x, additive, label='additive')
plt.plot(x, multiplicative, label='multiplicative')
plt.grid()
plt.xlabel('x')
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_14_0.png)
    


Il modello che fa al caso nostro è quello additivo.

# Simple Data Seasonal Decomposition


```python
df_data = df_data.fillna(value=df_data.mean())
data = df_data[sensors[0]].to_numpy()

data_decomposition = seasonal_decompose(data, model='additive', period=period, filt=[1/period for _ in range(period)])
```

Scomponiamo il segnale di misura in trend, pattern e residuo. Scegliamo una finestra temporale di 24 ore sia per la media mobile (trend) che per la stagionalità (seasonal)


```python
data_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_19_0.png)
    



```python
df_tmp = df_tmp.fillna(value=df_tmp.mean())
tmp = df_tmp[tmp_sensors].mean(axis=1).to_numpy()

tmp_decomposition = seasonal_decompose(tmp, model='additive', period=period, filt=[1/period for _ in range(period)])
```

Facciamo lo stesso per la temperatura


```python
tmp_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_22_0.png)
    


Confrontiamo tra loro i trend di misure temperatura 


```python
plt.plot(df_data['time'], data_decomposition.trend, label='data', color='blue', alpha=0.6)
plt.grid()
plt.xticks(rotation=20)
plt.xlabel('Time')
plt.legend(loc='upper left')

plt.twinx()
plt.plot(df_tmp['time'], tmp_decomposition.trend, label='temperature', color='red', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_24_0.png)
    


L'algoritmo visto finora è molto semplice e rudimentale. Per capirne le limitazioni, soprattutto nell'individuazione del pattern periodico, vediamolo più nel dettaglio.  
La ripetizione periodica del pattern viene ricercata nel seguente modo:
- La serie temporale viene suddivisa in blocchi del periodo desiderato
- I blocchi vengono tutti "sovrapposti"
- Si mediano i vari blocchi sovrapposti
- Il "blocco medio" così ottenuto viene replicato N volte in modo da coprire la serie temporale originaria

Vediamo un esempio esplicito: dividiamo la serie temporale in blocchi di durata *period* (24 ore) e visualizziamo i primi 6 blocchi. 


```python
s = sensors[0]
N = len(df)
blocks = []
t = [i * dt /3600 for i in range(period)]
for idx, block in df.groupby(np.arange(N) // period):
    plt.subplot(3,3,idx+1)
    plt.plot(t, block[s].values)
    plt.grid()
    if idx == 5:
        break
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_26_0.png)
    


Poi calcoliamo il blocco medio mediando sui 6 blocchi.


```python
s = sensors[0]
blocks = []
for idx, block in df.groupby(np.arange(len(df)) // period):
    plt.plot(t, block[s].values, label=f'block {idx+1}', color='royalblue', alpha=0.3)
    blocks.append(block[s].values)
    if idx == 5:
        break

blocks = np.stack(blocks)
avg_block = blocks.mean(axis=0)

plt.plot(t, avg_block, color="black", linewidth=2)
plt.grid()
plt.gca().set_facecolor((0.75, 0.75, 0.75, 0.4))
plt.title('Blocco medio (1 giorno)')
plt.xlabel('Tempo [ore]')
plt.show()
```


    
![png](eda_files/eda_28_0.png)
    


La componente stagionale del segnale originale sarà quindi costruita ripetendo la curva nera in figura fino a coprire l'intero arco temporale.  
La componente di trend, invece, viene ottenuta con una media mobile.

Questo algoritmo di decomposizione è ottimo per acquisire intuizione sul funzionamento della decomposizione seasonal-trend di un segnale, ma presenta anche forti limiti. Ne elenchiamo di seguito alcuni, con particolare riferimento ai nostri scopi:

# STL Decomposition  
### Seasonal-Trend Decomposition using LOESS


```python
seasonal = 30 * period # period rate of change  (LOESS)
seasonal = seasonal if seasonal % 2 else seasonal + 1 # make sure it's odd

data_decomposition = STL(
    endog=data, # measurements
    period=period, # period (periodic pattern)
    seasonal=seasonal, # period rate of change  (LOESS)
    trend=2*period+1 # smoothing window length (LOESS)
).fit()
data_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_32_0.png)
    



```python
tmp_decomposition = STL(
    endog=tmp, # measurements
    period=period, # period (periodic pattern)
    seasonal=seasonal, # period rate of change  (LOESS)
    trend=2*period+1 # smoothing window length (LOESS)
).fit()
tmp_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_33_0.png)
    



```python
plt.plot(df_data['time'], data_decomposition.trend, label='data', color='blue', alpha=0.6)
plt.grid()
plt.xticks(rotation=20)
plt.xlabel('Time')
plt.legend(loc='upper left')

plt.twinx()
plt.plot(df_tmp['time'], tmp_decomposition.trend, label='temperature', color='red', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_34_0.png)
    


# MSTL Decomposition  
### Multi-Seasonal-Trend Decomposition using LOESS


```python
seasonal = 30 * period # period rate of change  (LOESS)
seasonal = seasonal if seasonal % 2 else seasonal + 1 # make sure it's odd

data_decomposition = MSTL(
    endog=data, # measurements
    periods=[period, 7*period, 365*period], # period (periodic pattern)
    windows=[2*period+1, 2*7*period+1, 2*365*period+1],
    lmbda=None,
    stl_kwargs={'trend':2*7*period+1}
).fit()
data_decomposition.plot()
plt.show()
```

    c:\Users\m.zortea\Documents\timeseries_etl\.venv\Lib\site-packages\statsmodels\tsa\stl\mstl.py:218: UserWarning: A period(s) is larger than half the length of time series. Removing these period(s).
      warnings.warn(
    


    
![png](eda_files/eda_36_1.png)
    



```python
tmp_decomposition = MSTL(
    endog=tmp, # measurements
    periods=[period, 7*period, 365*period], # period (periodic pattern)
    windows=[2*period+1, 2*7*period+1, 2*365*period+1],
    lmbda=None,
    stl_kwargs={'trend':2*7*period+1}
).fit()
tmp_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_37_0.png)
    



```python
plt.plot(df_data['time'], data_decomposition.trend, label='data', color='blue', alpha=0.6)
plt.grid()
plt.xticks(rotation=20)
plt.xlabel('Time')
plt.legend(loc='upper left')

plt.twinx()
plt.plot(df_tmp['time'], tmp_decomposition.trend, label='temperature', color='red', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_38_0.png)
    



```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
tmp_decomposition.trend = np.array([el for el in tmp_decomposition.trend])
model.fit(tmp_decomposition.trend.reshape(-1,1), data_decomposition.trend)
plt.plot(df_data['time'], model.predict(tmp_decomposition.trend))
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[46], line 4
          1 from sklearn.linear_model import LinearRegression
          3 model = LinearRegression()
    ----> 4 tmp_decomposition.trend = np.array([el for el in tmp_decomposition.trend])
          5 model.fit(tmp_decomposition.trend.reshape(-1,1), data_decomposition.trend)
          6 plt.plot(df_data['time'], model.predict(tmp_decomposition.trend))
    

    AttributeError: property 'trend' of 'DecomposeResult' object has no setter


# Algorithm Test

Riusciamo ad individuare un salto nei dati non dovuto alla temperatura?  
Aggiungiamo artificialmente un salto del 5% nei dati a circa 2/3 della serie temporale di


```python
N = len(data)
data[2*N//3:] -= 3.0
```


```python
seasonal = 30 * period # period rate of change  (LOESS)
seasonal = seasonal if seasonal % 2 else seasonal + 1 # make sure it's odd

data_decomposition = MSTL(
    endog=data, # measurements
    periods=[period, 7*period, 365*period], # period (periodic pattern)
    windows=[2*period+1, 2*7*period+1, 2*365*period+1],
    lmbda=None,
    stl_kwargs={'trend':2*7*period+1}
).fit()
data_decomposition.plot()
plt.show()
```

    c:\Users\m.zortea\Documents\timeseries_etl\.venv\Lib\site-packages\statsmodels\tsa\stl\mstl.py:218: UserWarning: A period(s) is larger than half the length of time series. Removing these period(s).
      warnings.warn(
    


    
![png](eda_files/eda_43_1.png)
    



```python
tmp_decomposition = MSTL(
    endog=tmp, # measurements
    periods=[period, 7*period, 365*period], # period (periodic pattern)
    windows=[2*period+1, 2*7*period+1, 2*365*period+1],
    lmbda=None,
    stl_kwargs={'trend':2*7*period+1}
).fit()
tmp_decomposition.plot()
plt.show()
```


    
![png](eda_files/eda_44_0.png)
    



```python
plt.plot(df_data['time'], data_decomposition.trend, label='data', color='blue', alpha=0.6)
plt.grid()
plt.xticks(rotation=20)
plt.xlabel('Time')
plt.legend(loc='upper left')

plt.twinx()
plt.plot(df_tmp['time'], tmp_decomposition.trend, label='temperature', color='red', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](eda_files/eda_45_0.png)
    



```python

```
