import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression

nskip = 213

df = pd.read_csv("../grain_bound_sims/log.lammps", skiprows=nskip,
                 delim_whitespace=True)

steps = df["Step"].to_numpy()
msd = df["v_msd_fm"].to_numpy()

plt.plot(steps, msd)
plt.xlabel("Steps")
plt.ylabel("MSD (fm)")
plt.savefig("./lj_run_8_20_msd_plot.png")

# generating fit
model = LinearRegression()
model.fit(steps.reshape(1, -1),msd)

print(model.coef_, model.intercept_)

