import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

epochs = 20
loss = np.random.rand(20)
test_acc = np.random.rand(20) * 0.0001

plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
x = range(epochs)
plt.subplot(121);
plt.scatter(x,loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss decline")

plt.subplot(122);
plt.plot(x,test_acc)
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("test accuracy")
plt.show()
