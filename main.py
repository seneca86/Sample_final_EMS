# Ex 1
# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from sklearn.inspection import plot_partial_dependence
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pathlib import Path
from pmdarima.arima import StepwiseContext

# %%
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# %%
red_wine = pd.read_csv('.lesson/assets/winequality-red.csv', sep=';')
# %%
red_wine['quality'].value_counts()
# %%
plt.hist(red_wine[['quality', 'alcohol']], rwidth=1, bins=10, label='quality')
plt.savefig(directory + 'quality_alcohol_hist.png')
plt.clf()
# %%
formula = 'quality ~ alcohol'
model = smf.ols(formula, data = red_wine)
results = model.fit()
results.summary()
# %%
inter = results.params['Intercept']
slope = results.params['alcohol']
print(f'{inter=}')
print(f'{slope=}')
# %%
plt.scatter(x='alcohol',y='quality', c='pH', data=red_wine)
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.legend()
plt.savefig(directory + 'quality_alcohol_scatter.png')
plt.clf()
# %%
formula = 'quality ~ alcohol + pH + chlorides + density'
model = smf.ols(formula, data = red_wine)
results = model.fit()
results.summary()
inter = results.params['Intercept']
slope = results.params['alcohol']
print(f'{inter=}')
print(f'{slope=}')
# %%
red_wine['category'] = 'red'
white_wine = pd.read_csv('.lesson/assets/winequality-white.csv', sep=';')
white_wine['category'] = 'white'
# %%
wine = pd.concat([white_wine, red_wine])
# %%
plt.boxplot(x=[red_wine.quality, white_wine.quality])
plt.xlabel('category')
plt.ylabel('quality')
plt.legend()
plt.savefig(directory + 'category_wine_boxplot.png')
plt.clf()
# %%
wine['red'] = (wine['category'] == 'red') * 1
formula = 'red ~ alcohol + pH + chlorides + density'
model = smf.logit(formula, data=wine)
results = model.fit()
results.summary()
# %%
new = pd.DataFrame([[11, 3.3, 0.06, 1]], columns=['alcohol', 'pH', 'chlorides', 'density'])
y = results.predict(new)
print(f'The chances of this wine being red are {y[0]}')

# Exercise 2
# %%
eeg = pd.read_csv('.lesson/assets/plrx.txt', sep='\t')
var = 'col1'
ix = 1
# %%
plt.plot(eeg.iloc[:,ix])
plt.legend()
plt.savefig(directory + '/actuals_eeg')
plt.clf()
# %%
plot_acf(eeg[var], lags=20)
plt.legend()
plt.savefig(directory + '/acf')
plt.clf()
plot_pacf(eeg[var], lags=20)
plt.savefig(directory + '/pacf')
plt.clf()

# %%
split = 175
train = eeg.iloc[0:split, ix]
test = eeg.iloc[split:, ix]
# %%
with StepwiseContext(max_dur=15):
    model = pm.auto_arima(
        train,
        stepwise=True,
        error_action="ignore",
        seasonal=True,
    )
# %%
print(f"{model.summary()}")
preds, conf_int = model.predict(n_periods=eeg.shape[0] - split, return_conf_int=True)
# %%
plt.plot(train, label="actuals", color="black")
plt.plot(range(split,split + len(test)), preds, label="autoarima", color="blue")
plt.legend()
plt.savefig(directory + "/autoarima")
plt.clf()

# Exercise 3
# %%
