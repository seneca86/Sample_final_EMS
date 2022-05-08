# Ex 1
# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
import pmdarima as pm
from pathlib import Path
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
# %%
plt.plot(eeg.iloc[:,1])
# %%

