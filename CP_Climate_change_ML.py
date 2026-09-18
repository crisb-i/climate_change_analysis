import pandas as pd

climate_df = pd.read_csv(r"C:\Users\Chris\Documents\Projects\Climate_Model\Climate_Change\climate_change_data.csv")

print(climate_df.head(10));

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


DATA_PATH = os.path.join(os.path.dirname(__file__), "climate_change_data.csv")


def run_regression_analysis():
    climate_df = pd.read_csv(DATA_PATH)
    climate_df["Date"] = pd.to_datetime(climate_df["Date"])

    monthly_df = (
        climate_df.set_index("Date")[["Temperature", "CO2 Emissions"]] # Resample to monthly frequency
        .resample("MS")
        .mean()
        .dropna()
        .reset_index()
    )

    X = monthly_df[["CO2 Emissions"]]
    y = monthly_df["Temperature"]

    model = LinearRegression()
    model.fit(X, y)

    r_squared = model.score(X, y) # overall it appears this does not have a strong correlation and the data honestly doesnt fit a linear regression model well
    slope = model.coef_[0]
    intercept = model.intercept_
    correlation = monthly_df["CO2 Emissions"].corr(monthly_df["Temperature"])
    p_value = linregress(monthly_df["CO2 Emissions"], monthly_df["Temperature"]).pvalue

    print("Monthly Regression Analysis: Carbon Emissions (Independent Variable) vs Temperature (Dependent Variable)")
    print("-" * 80)
    print(f"Original samples: {len(climate_df)}")
    print(f"Monthly samples analyzed: {len(monthly_df)}")
    print(f"Model equation: Temperature = {intercept:.6f} + ({slope:.12f} * CO2 Emissions)")
    print(f"Coefficient (slope): {slope:.12f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R-squared: {r_squared:.12f}")
    print(f"Correlation coefficient: {correlation:.12f}")
    print(f"P-value for slope: {p_value:.12f}")
    print()

    plt.figure(figsize=(10, 6))
    sorted_monthly_df = monthly_df.sort_values("CO2 Emissions")
    sorted_predictions = model.predict(sorted_monthly_df[["CO2 Emissions"]])
    plt.scatter(monthly_df["CO2 Emissions"], monthly_df["Temperature"], alpha=0.7, color="steelblue")
    plt.plot(
        sorted_monthly_df["CO2 Emissions"],
        sorted_predictions,
        color="darkorange",
        linewidth=2,
        label="Regression Line",
    )
    plt.title("Monthly Regression of Temperature on CO2 Emissions")
    plt.xlabel("CO2 Emissions")
    plt.ylabel("Temperature")
    plt.text(
        0.05,
        0.95,
        f"$R^2 = {r_squared:.6f}$",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "monthly_regression_plot.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    run_regression_analysis()
    
