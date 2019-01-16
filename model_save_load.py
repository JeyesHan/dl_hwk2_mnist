from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('test.png')
img = np.array(img)
if img.ndim == 3:
    img = img[:,:,0]
img = np.random.randint(0,255,(628,800))
print (img.shape)
loss = [1,5,7]
test_acc = [5,6,8]
plt.subplot(121);
plt.plot(loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss decline")

plt.subplot(122);
plt.plot(test_acc)
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("test accuracy")
plt.show()

