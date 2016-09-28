def plot(model,X):
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import stats
    
    xmin, ymin = X.min(0)
    xmax, ymax = X.max(0)
    xmin, xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
    ymin, ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T

    plt.figure()
    zz = model.predict(Xgrid).reshape(100, 100)
    zz_data = model.predict(X)

    plt.contour(xx, yy, zz, [stats.scoreatpercentile(zz_data, 5)], colors='k', linewidths=3)
    plt.scatter(X[:,0], X[:,1], 30, np.argmax(model.get_phi(), 1), linewidth=0, cmap=plt.cm.gist_rainbow)

    zz_components = model.predict_components(Xgrid)
    phi_hat = model.get_phihat()
    pie = phi_hat+model.alpha
    pie /= pie.sum()
    zz_components *= pie[np.newaxis,:]
    [plt.contour(xx, yy, zz.reshape(100, 100), [stats.scoreatpercentile(zz_data, 5.)], colors='k', linewidths=1)\
        for zz in zz_components.T]
    plt.show()
