import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import t

@st.cache_data
def load_data():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")
    df = pd.concat([X, y], axis=1)
    return X, df

X, df = load_data()
st.header('Linear Regression From Scratch!')
st.write('So first off, the goal for this is to use basic mathematics to perform all of the calculations')
st.write('That means no packges that do most of the work for you')

st.write("First we're just going to graph one variable against the target variable")
regressor = st.selectbox("Select variable:", X.columns, index=0)

x = df[regressor]
y = df['target']


x_mean = x.mean()
y_mean = y.mean()

numerator = 0
denominator = 0

for i in range(len(x)):
    numerator = numerator + (x[i]-x_mean)*(y[i]-y_mean)
    denominator = denominator + (x[i]-x_mean)**2

slope = numerator/denominator
y_int = y_mean - slope*x_mean

y_pred = [slope*xi+y_int for xi in x]

st.header('Slope')
st.write('The way that the slope is calculated uses:')
st.latex(r"m=\frac{\Sigma (x_i-\bar{x})(y_i-\bar{y})}{\Sigma (x_i-\bar{x})^2}")
st.write(f"Slope: {slope:.3f}")

st.header('y-intercept')
st.write('The y-intercept is calculated using:')
st.latex(r'b=\bar{y}-m\bar{x}')
st.write(f"y-intercept: {y_int:.3f}")

st.header('Final Equation')
st.write(f"Equation: y= {slope:.3f} x + {y_int:.3f}")

fig, ax = plt.subplots()
ax.scatter(x, y, label="Data points")
ax.plot(x, y_pred, color='red', label='Line of Best Fit')
ax.set_xlabel(regressor)
ax.set_ylabel("Target")
ax.legend()

st.pyplot(fig)

st.header('R-Squared')

ssr = 0
sst = 0
for i in range(len(y)):
    ssr = ssr + (y[i]-y_pred[i])**2
    sst = sst + (y[i]-y_mean)**2

st.write("Next we'll calculate two different things: SSR & SST")
st.write("- SSR: Residual Sum of Squared")
st.latex(rf"SSR=\Sigma ((y_i-\hat{{y}})^2)={ssr:.3f} \qquad \hat{{y}}=y_{{pred}}")

st.write("- SST: Total Sum of Squared")
st.latex(rf"SST=\Sigma ((y_i-\bar{{y}})^2)={sst:.3f} \qquad \bar{{y}}=y_{{mean}}")

st.write("- R2: Coefficient of Determination")
r2 = 1-ssr/sst
st.latex(rf"R^2=1-\frac{{SSR}}{{SST}}={r2:.3f}")
st.write("Essentially, if R2 is close to 0, there is no correlation. If it's close to -1 or 1 there is almost a perfect correlation between the independent variable and the dependent variable.")




st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

st.header('Multiple Linear Regression')



# Adding an intercept
X_with_intercept = X.copy()
X_with_intercept.insert(0, "Intercept", 1.0)

st.write("Now we're going to switch from using one variable, to all the variables")
st.write("For this we'll switch to using matrix algebra")

Xt = X_with_intercept.T
XtX = Xt @ X_with_intercept
XtX_inv = np.linalg.inv(XtX)
Xt_y = Xt @ y
beta = XtX_inv @ Xt_y

st.latex(r"\beta:\, The\, matrix\, of\, slopes")
st.latex(r"X:\, The\, matrix\, of\, regressors")
st.latex(rf"\beta=(X^TX)^{{-1}}X^Ty")

st.write("Next we want to determine how significant each variable is")
st.write("For this we'll use standard erros, t-scores and their associated p-values")

y_hat = X_with_intercept @ beta
residuals = y - y_hat
n = len(y)
p = X_with_intercept.shape[1]

rss = np.sum(residuals**2)
sigma_squared = rss / (n - p)

var_beta = sigma_squared * XtX_inv

standard_errors = np.sqrt(np.diag(var_beta))
t_scores = beta / standard_errors

p_values = 2 * (1 - t.cdf(np.abs(t_scores), df=(n-p)))

st.write("- Standard Errors: How precisely a coefficient is from zero")
st.write("(The calculation is too long, not going to include for now)")

st.write("- t-Score: How far a standard error is from zero")
st.latex(r"t_j = \frac{{\beta_j}}{{SE{{(\beta_j)}}}} = \frac{{\beta_j}}{{\sqrt{{Var(\beta_j)}}}}")

st.write("- p-value: Different interpretation of the t-value, will consider the variable significant enough if below 0.05")
st.write("Calculation of this involves referencing a table (you'll remember this from high school)")

results_df = pd.DataFrame({
    "Feature": X_with_intercept.columns,
    "Coefficients (beta)": beta,
    "Standard Error": standard_errors,
    "t-score": t_scores,
    "p-value": p_values
})
st.table(results_df)


y_mean = np.mean(y)

ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)

r_squared = 1 - (ss_res / ss_tot)

st.header("Resulting R2")
st.metric(label="R² Score", value=f"{r_squared:.4f}")


st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

selected_features = {}
st.write("### Regression Coefficient Table")
st.write("Okay, so the last part")
st.write("Now we can run the regression excluding specific variables (drop the ones with highest p-values)")


# Assume results_df contains your table (with Intercept already included)

col1, col2 = st.columns([3, 1])  # Adjust width ratios as needed

with col1:
    st.write("### Regression Results")
    st.table(results_df)  # static table with all rows

with col2:
    st.write("### Include Variables")
    selected_features = []
    for feature in results_df["Feature"]:
        if feature == "Intercept":
            continue  # always included
        include = st.checkbox(f"Include '{feature}'", value=True, key=f"chk0_{feature}")
        if include:
            selected_features.append(feature)

# If at least one feature is selected:
if selected_features:
    X_selected = X[selected_features].copy()
    X_selected.insert(0, "Intercept", 1.0)
    feature_names = X_selected.columns

    # Recalculate regression
    X_np = X_selected.to_numpy()
    y_np = y.to_numpy() if isinstance(y, pd.Series) else y
    beta = np.linalg.inv(X_np.T @ X_np) @ (X_np.T @ y_np)

    y_hat = X_np @ beta
    n = len(y_np)
    p = X_np.shape[1]
    residuals = y_np - y_hat

    # Residual variance
    sigma_squared = np.sum(residuals**2) / (n - p)
    XtX_inv = np.linalg.inv(X_np.T @ X_np)
    se = np.sqrt(np.diag(sigma_squared * XtX_inv))
    t_scores = beta / se
    p_values = 2 * (1 - t.cdf(np.abs(t_scores), df=n - p))

    # R²
    ss_res = np.sum((y_np - y_hat)**2)
    ss_tot = np.sum((y_np - np.mean(y_np))**2)
    r_squared = 1 - ss_res / ss_tot

    # Output updated table
    updated_results = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": beta,
        "Std. Error": se,
        "t-score": t_scores,
        "p-value": p_values
    })

    st.write("### Updated Regression Results")
    st.table(updated_results)
    st.metric(label="R² Score", value=f"{r_squared:.4f}")

else:
    st.warning("Please select at least one feature to include in the model.")
