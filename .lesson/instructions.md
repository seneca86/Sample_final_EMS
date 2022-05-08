# Sample Final Economic Modeling and Simulation

* Important: the midterm is different from the sample in significant ways, so if you just copy-paste the code your solutions will be most likely wrong and your grade will be penalized. If I detect errors that show that you have __copied from another classmate, your grade will me penalized__ even more.

* The maximum score is 100 points, and you can get an additional 15 pts if your code is __particulary clean and original__; in other words, you can score less than 100 point in the exercises and still get a 100.

* This exam is __open-book__: you may also lookup on the internet as long as you do not communicate with your classmates or anyone else.

* The total amount of points is 100, but you can get up to 15 bonus points if your code is clean and/or elegant. In other words, you do not need to get everything right to get the maximum grade. However, it is __critical that the code runs__ and that there are no execution errors: a code that runs but that misses some calculations will be graded benevolently; a code that does not run will not.

* The midterm involves quite a lot of __plotting__; I have used the library `matplotlib.pyplot` extensively in the sample midterm because I find it easier for you to apply to time series, compared to `seaborn`. However, you are of course free to use `seaborn` if you prefer.

* For the exercises below you will need the following libraries, parameters, and code to create the path for the plots:

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
import pmdarima as pm
```

```python
from pathlib import Path
```

```python
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200
```

```python
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)
```

## Exercise 1: [45 pts]

(1) [5 pts] Load the dataset stored in `.lesson/assets/winequality-red.csv`. The separator is a semicolon.

(2) [5 pts] Count the number of occurrences of each value of `quality`, i.e. how many red wines score quality 0, how many score quality 1, and so on.

(3) [5 pts] Plot a histogram with the distributions of `quality` and `alcohol`.

(4) [5 pts] Build a linear regression where `quality` is the `y` variable and `alcohol` the `x` variable; print a summary of the results, and the intercept and the slope.

(5) [5 pts] Build a scatterplot where `alcohol` is the `x` variable and `quality` the `y` variable; include `pH` as the color.

(6) [5 pts] Build another linear regression with the regressors `alcohol`, `pH`, `chlorides`, and `density` in order to predict the `quality`. Print the intercept and slope of `alcohol`.

(7) [5 pts] Load the dataset stored in `.lesson/assets/winequality-white.csv`. Concatenate it to the dataset we read in (1).

(8) [5 pts] Build a logistic regression that predicts whether a wine is red or white based on the same regressors we used in (6). You will need to generate a binary variable named `red`.

(9) [5 pts] Estimate the probabilities that a wine is red given its `alcohol` is 11, its `pH` is 3.3, its `chlorides` are 0.06, and its `density` is 1.

## Exercise 2: []

