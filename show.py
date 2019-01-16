import numpy as np
import matplotlib.pyplot as plt


#1*20*24*24
a = np.random.rand(16,16,1)
a = a*255
a = a.astype('uint8')


def show_feature_map(x,size,rows,colums):#assume x is 1*20*24*24 size=24 rows=4 colums=5
    res = np.zeros((size * rows, size * colums))
    for i in range(rows):
        for j in range(colums):
            temp = np.squeeze(x[:, colums * i + j,:,:])
            res[i*size:(i+1)*size,j*size:(j+1)*size] = temp
    #print (res)
    #plt.imshow(res,cmap = 'Greys')
    plt.imshow(res, cmap='gray')
    plt.axis("off")
    plt.show()

def norm_feature_map(x):
    min = x.min()
    max = x.max()
    k = 255/(max-min)
    x = (x-min)*k
    x = x.astype('uint8')
    return x

#x = (np.random.rand(1,20,24,24)<-1) *255
#x = np.random.randint(0,255,(1,20,24,24))
#print (x)

#x = x.astype('uint8')
#print (x)
#show_feature_map(x,24,4,5)
#plt.imshow(x[0,0,:,:],cmap='Greys')
#plt.show()

x = np.random.randn(1,20,24,24)
print (x)
show_feature_map(x,24,4,5)

y = norm_feature_map(x)
print (y)
show_feature_map(y,24,4,5)

