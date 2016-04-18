import matplotlib.pyplot as plt
import pickle
import numpy as np

TIME = 1000
INTERVAL = 50
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
plt.errorbar(x, data32, yerr=err32)
plt.errorbar(x, data43, yerr=err43)
plt.errorbar(x, data43t, yerr=err43t)
plt.legend(['32', '43', '43t'])
plt.show()