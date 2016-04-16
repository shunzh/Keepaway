import pickle
import matplotlib.pyplot as plt

t = pickle.load(open('time.p'))
plt.plot(t)
plt.show()

w = pickle.load(open('weights.p'))
print w