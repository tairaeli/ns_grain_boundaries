import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression

nskip = 209

df = pd.read_csv("../grain_bound_sims/log.lammps", skiprows=nskip,
                 delim_whitespace=True)

steps = df["Step"].to_numpy()
msd = df["v_msd_fm"].to_numpy()

plt.plot(steps, msd)
plt.xlabel("Steps")
plt.ylabel("MSD (fm)")
plt.savefig("msd_plot.png")

# generating fit
model = LinearRegression()
model.fit(steps.T,msd)

print(model.coef_, model.intercept_)

