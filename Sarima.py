import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams

##### Leer datos 
df=pd.read_csv('Pronostico 3.csv',sep=';')[['Valor Neto',' Semana del año ',' FECHA completa ']]
df.head()
df.columns
##### Agregar datos por semana
semana=2
valor=0
valores=[]
semanas_index=[]
for i in range(len(df)):
    semana_data=df.loc[i][' Semana del año ']
    if semana_data==semana:
        valor+=df.loc[i]['Valor Neto']
    else:
        valores.append(valor)
        valor=0
        semanas_index.append(df.loc[i][' FECHA completa '])
        semana=semana_data 
y={'valores':valores}
y=pd.DataFrame(y)
print(y.head())
y.index=pd.DatetimeIndex(freq='w',start='2015-01-01',periods=251)
y.head()

      

#### Graficar Datos para identificar tendencia 
y.plot(figsize=(19, 4))
plt.show()


##### Descomponiendo los datos
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#SARIMA to time series forecasting
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 53) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 0, 53),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(10, 8))
plt.show()
plt.savefig('books_read.png')

pred = results.get_prediction(start=pd.to_datetime('2019-03-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2019-03-31':]
mse = ((y_forecasted - y_truth) ** 2)
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

y_truth
y_forecasted
mse
