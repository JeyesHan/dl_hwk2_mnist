from __future__ import print_function
import math
import matplotlib.pyplot as plt
import numpy as np

def am(f1=1000,f2=100,a=1,direct=2,num=160,T=1):
    output = []
    for i in range(0,num):
        x = (1./f2/(num/T)) * i
        #x = i
        output.append((direct + a * math.cos(2*math.pi*f2*x)) * math.cos(2*math.pi*f1*x))
    return output

if __name__ == '__main__':

    am = am()
    x = np.arange(0,len(am))
    plt.plot(x, am)
    plt.title('AM')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    for item in am:
        print("{:.0f}".format((item+3)/6 * 255),end=',')
