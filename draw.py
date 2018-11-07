import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1, 11)


ga_ROC = np.array([[0.,0.,0.1146,0.1354,0.1562,0.1875,0.2812,0.3438,0.5,0.6354,1.],
                   [0.,0.,0.8567,0.8981,0.9172,0.9236,0.965 ,0.9745,0.9873,0.9904,1.]])
ga_SE = np.array([[1.,0.9904,0.9873,0.9745,0.965,0.9236,0.9172,0.8981,0.8567,0.7038,0.]])
ga_SP = np.array([[0.,0.3646,0.5,0.6562,0.7188,0.8125,0.8438,0.8646,0.8854,1.,1.]])
ga_ACC= np.array([[1.6354,2.1625,2.4189,2.6934,2.7761,2.6667,2.6934,2.5887,2.3289,1.6772,0.2341]])

PCA5_ROC = np.array([[0.,0.0208,0.0417,0.0938,0.1354,0.1771,0.2083,0.2604,0.3854,0.4896,1.],
                     [0.,0.8439,0.9013,0.9299,0.9459,0.9586,0.9745,0.9841,0.9904,0.9904,1.]])
PCA5_SE = np.array([[1.,0.9904,0.9904,0.9841,0.9745,0.9586,0.9459,0.9299,0.9013,0.8439,0.]])
PCA5_SP = np.array([[0.,0.5104,0.6146,0.7396,0.7917,0.8229,0.8646,0.9062,0.9583,0.9792,1.]])
PCA5_ACC = np.array([[1.6354,2.4658,2.7206,3.0159,3.0806,3.0159,3.0159,2.9843,2.8626,2.4422,0.2341]])

PCA4_ROC = np.array([[0.,0.,0.0625,0.9792,1.,1.,1.,1.,1.,1.,1.],
                     [0.,0.,0.051,0.9904,1.,1.,1.,1.,1.,1.,1.]])
PCA4_SE = np.array([[1.,1.,1.,1.,1.,1.,1.,0.9904,0.051,0.,0.]])
PCA4_SP = np.array([[0.,0.,0.,0.,0.,0.,0.,0.0208,0.9375,1.,1.]])
PCA4_ACC = np.array([[1.6354,1.6354,1.6354,1.6354,1.6354,1.6354,1.6354,1.6218,0.265,0.2341,0.2341]])

origin_ROC = np.array([[0.,0.0104,0.0521,0.0938,0.1354,0.1875,0.2083,0.2708,0.3958,0.5104,1.],
                       [0.,0.8408,0.8885,0.9236,0.9395,0.9618,0.9713,0.9841,0.9904,0.9904,1.]])
origin_SE = np.array([[1.,0.9904,0.9904,0.9841,0.9713,0.9618,0.9395,0.9236,0.8885,0.8408,0.]])
origin_SP = np.array([[0.,0.4896,0.6042,0.7292,0.7917,0.8125,0.8646,0.9062,0.9479,0.9896,1.]])
origin_ACC = np.array([[1.6354,2.4189,2.6934,2.9843,3.048,3.0159,2.9531,2.9225,2.7206,2.4422,0.2341]])



plt.scatter(ga_ROC[0,:], ga_ROC[1,:], marker='*', c= 'b')
plt.scatter(PCA5_ROC[0,:], PCA5_ROC[1,:], marker='x', c= 'g')
plt.scatter(PCA4_ROC[0,:], PCA4_ROC[1,:], marker='o', c= 'r')
plt.scatter(origin_ROC[0,:], origin_ROC[1,:], marker='>', c= 'k')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['GA','PCA5','PCA4', 'ORIGIN'], loc = 'lower right')
plt.plot(ga_ROC[0,:], ga_ROC[1,:], 'b')
plt.plot(PCA5_ROC[0,:], PCA5_ROC[1,:], 'g')
plt.plot(PCA4_ROC[0,:], PCA4_ROC[1,:], 'r')
plt.plot(origin_ROC[0,:], origin_ROC[1,:], 'k')
plt.title('ROC')
plt.show()

plt.scatter(x, ga_SE, marker='*', c= 'b')
plt.scatter(x, PCA5_SE, marker='x', c= 'g')
plt.scatter(x, PCA4_SE, marker='o', c= 'r')
plt.scatter(x, origin_SE, marker='>', c= 'k')
plt.xlabel('threshold')
plt.ylabel('SE')
plt.legend(['GA','PCA5','PCA4', 'ORIGIN'], loc = 'lower left')
plt.plot(x, ga_SE[0,:], 'b')
plt.plot(x, PCA5_SE[0,:], 'g')
plt.plot(x, PCA4_SE[0,:], 'r')
plt.plot(x, origin_SE[0,:], 'k')
plt.title('SE')
plt.show()

plt.scatter(x, ga_SP, marker='*', c= 'b')
plt.scatter(x, PCA5_SP, marker='x', c= 'g')
plt.scatter(x, PCA4_SP, marker='o', c= 'r')
plt.scatter(x, origin_SP, marker='>', c= 'k')
plt.xlabel('threshold')
plt.ylabel('SP')
plt.legend(['GA','PCA5','PCA4', 'ORIGIN'], loc = 'lower right')
plt.plot(x, ga_SP[0,:], 'b')
plt.plot(x, PCA5_SP[0,:], 'g')
plt.plot(x, PCA4_SP[0,:], 'r')
plt.plot(x, origin_SP[0,:], 'k')
plt.title('SP')
plt.show()

plt.scatter(x, ga_ACC, marker='*', c= 'b')
plt.scatter(x, PCA5_ACC, marker='x', c= 'g')
plt.scatter(x, PCA4_ACC, marker='o', c= 'r')
plt.scatter(x, origin_ACC, marker='>', c= 'k')
plt.xlabel('threshold')
plt.ylabel('ACC')
plt.legend(['GA','PCA5','PCA4', 'ORIGIN'], loc = 'lower left')
plt.plot(x, ga_ACC[0,:], 'b')
plt.plot(x, PCA5_ACC[0,:], 'g')
plt.plot(x, PCA4_ACC[0,:], 'r')
plt.plot(x, origin_ACC[0,:], 'k')
plt.title('ACC')
plt.show()
