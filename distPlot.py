import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns

x = np.random.standard_normal(1000)
b = np.random.standard_normal(1000) + max(x) - x.mean()

plt.figure()
sns.distplot(x, bins=50)
plt.legend(["Distribution of Training Data"])

plt.figure()
plt.plot(b.mean(), 0)
sns.distplot(b, bins=50)
plt.legend(["Distribution of Test/Deployed Data"])

plt.figure()
sns.distplot(x, kde = True, bins=50)
sns.distplot(b, kde= True, bins=50)

plt.legend(["Data In Distribution", "Data Outside Distribution"])
