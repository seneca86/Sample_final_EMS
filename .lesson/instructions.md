# Sample Final Economic Modeling and Simulation

* Important: the midterm is different from the sample in significant ways, so if you just copy-paste the code your solutions will be most likely wrong and your grade will be penalized. If I detect errors that show that you have __copied from another classmate, your grade will me penalized__ even more.

* The maximum score is 100 points, and you can get an additional 15 pts if your code is __particulary clean and original__; in other words, you can score less than 100 point in the exercises and still get a 100.

* This exam is __open-book__: you may also lookup on the internet as long as you do not communicate with your classmates or anyone else.

* The total amount of points is 100, but you can get up to 15 bonus points if your code is clean and/or elegant. In other words, you do not need to get everything right to get the maximum grade. However, it is __critical that the code runs__ and that there are no execution errors: a code that runs but that misses some calculations will be graded benevolently; a code that does not run will not.

* The midterm involves quite a lot of __plotting__; I have used the library `matplotlib.pyplot` extensively in the sample midterm because I find it easier for you to apply to time series, compared to `seaborn`. However, you are of course free to use `seaborn` if you prefer.

* Store the plots in files with `plt.savefig()`; clean the canvas after every plot with `plt.clf()`

* For the exercises below you will need the following libraries, parameters, and code to create the path for the plots:

```python
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

## Exercise 1: Exploratory analysis and regression [40 pts]

(1) [0 pts] Load the dataset stored in `.lesson/assets/winequality-red.csv`. The separator is a semicolon.

(2) [5 pts] Count the number of occurrences of each value of `quality`, i.e. how many red wines score quality 0, how many score quality 1, and so on.

(3) [5 pts] Plot a histogram with the distributions of `quality` and `alcohol`.

(4) [5 pts] Build a linear regression where `quality` is the `y` variable and `alcohol` the `x` variable; print a summary of the results, and the intercept and the slope.

(5) [5 pts] Build a scatterplot where `alcohol` is the `x` variable and `quality` the `y` variable; include `pH` as the color.

(6) [5 pts] Build another linear regression with the regressors `alcohol`, `pH`, `chlorides`, and `density` in order to predict the `quality`. Print the intercept and slope of `alcohol`.

(7) [5 pts] Load the dataset stored in `.lesson/assets/winequality-white.csv`. Concatenate it to the dataset we read in (1).

(8) [5 pts] Build a logistic regression that predicts whether a wine is red or white based on the same regressors we used in (6). You will need to generate a binary variable named `red`.

(9) [5 pts] Estimate the probabilities that a wine is red given its `alcohol` is 11, its `pH` is 3.3, its `chlorides` are 0.06, and its `density` is 1.

## Exercise 2: Statistical methods for time series [30 pts]

(1) [5 pts] Read the file `.lesson/assets/plrx.txt`; the separator is the tab keystroke `\t`

(2) [5 pts] We will use the time series in column `col1`. Plot the time series.

(3) [5 pts] Plot the autocorrelation function and the partial autocorrelation function.

(4) [5 pts] Split the time series into train (from element `0` until element `175`) and test (from element `175` until the end) set.

(5) [5 pts] Fit an autoarima model to the time series.

(6) [5 pts] Print the summary of the model, and plot the prediction together with the actuals.

## Exercise 3: Bayesian statistics [30 pts]


(1) [10 pts] M&M’s are small candy-coated chocolates that come in a variety of colors. Mars, Inc., which makes M&M’s, changes the mixture of colors from time to time. In 1995, they introduced blue M&M’s.

* In 1994, the color mix in a bag of plain M&M’s was 30% Brown, 20% Yellow, 20% Red, 10% Green, 10% Orange, 10% Tan.

* In 1996, it was 24% Blue , 20% Green, 16% Orange, 14% Yellow, 13% Red, 13% Brown.

Suppose a friend of mine has two bags of M&M’s, and he tells me that one is from 1994 and one from 1996. He won’t tell me which is which, but he gives me one M&M from each bag. One is yellow and one is green. What is the probability that the yellow one came from the 1994 bag?

(2) [10 pts] Suppose you meet someone and learn that they have two children. You ask if either child is a girl and they say yes. What is the probability that both children are girls?

(3) [10 pts] Load the `titanic` dataset from `seaborn`. Calculate the probability of survival conditional on being an adult male.