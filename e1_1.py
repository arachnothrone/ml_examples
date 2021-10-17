import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import sklearn
from sklearn.linear_model import LinearRegression


def prepare_country_stats(oecdbli, gdppc):
    oecdbli = oecdbli[oecdbli["INEQUALITY"] == "TOT"]
    oecdbli = oecdbli.pivot(index="Country", columns="Indicator", values="Value")
    gdppc.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdppc.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecdbli, right=gdppc,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# Load data
oecd_bli_path = "book_git_repo/handson-ml/datasets/lifesat/oecd_bli_2015.csv"
gpd_per_capita_path = "book_git_repo/handson-ml/datasets/lifesat/gdp_per_capita.csv"

oecd_bli = pd.read_csv(oecd_bli_path, thousands=',')
gdp_per_capita = pd.read_csv(gpd_per_capita_path, thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

# Prepare data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
# plt.show()

# Select a linear model
# lin_reg_model = sklearn.linear_model.LinearRegression()
lin_reg_model = LinearRegression()

# Train the model
lin_reg_model.fit(x, y)

# Make a prediction for Cyprus
x_new = [[22587]]   # Cyprus GDP per capita
print("Prediction for Cyprus life satisfaction: {}, (GDP = {})".format(lin_reg_model.predict(x_new), x_new))
