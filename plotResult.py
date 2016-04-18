import matplotlib.pyplot as plt
import pickle
import numpy as np

TIME = 300
INTERVAL = 20
K = 2

def loadData(filename, rang):
  data = []
  for i in rang:
    d = pickle.load(open(filename+str(i)+'.p'))
    data.append(d[:TIME])

  means = []
  errs = []
  for j in range(0, TIME, INTERVAL):
    d = [data[i][k] for i in rang for k in range(j + INTERVAL)]
    means.append(np.mean(d))
    errs.append(1.96 * np.std(d) / np.sqrt(len(d)))
  return means, errs

data32, err32 = loadData('time32', range(0, K))
data43, err43 = loadData('time43', range(0, K))
data43t, err43t = loadData('time43t', range(0, K))

x = range(0, TIME, INTERVAL)
plt.errorbar(x, data32, yerr=err32, fmt='+--')
plt.errorbar(x, data43, yerr=err43, fmt='x--')
plt.errorbar(x, data43t, yerr=err43t, fmt='*-')
plt.xlabel("Number of Iterations")
plt.ylabel("Accumulated Rewards")
plt.xlim([-20, 1000])
plt.legend(['3 vs. 2', '4 vs. 3', '4 vs. 3 with transfer'], 'lower right')
plt.show()