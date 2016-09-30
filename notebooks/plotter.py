import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def MOHGPplot_simple(model,X,Y):
        assert X.shape[1]==1, "can only plot mixtures of 1D functions"
        #plt.figure()
        plt.plot(Y.T,'k',linewidth=0.5,alpha=0.4)
        mu, var = model.predict_components(X)
        plt.plot(mu[:,model.get_phihat()>1e-3],'k',linewidth=2)
        plt.show()


def MOHGPplot(model, X, Y, on_subplots=True, colour=False, newfig=True, errorbars=False,
    in_a_row=False, joined=True, gpplot=True,min_in_cluster=1e-3, data_in_grey=False, 
    numbered=True, data_in_replicate=False, fixed_inputs=[], ylim=None):
        """
        Plot the mixture of Gaussian processes. Some of these arguments are rather esoteric! The 
        defaults should be okay for most cases.
        Arguments
        ---------
        on_subplots (bool) whether to plot all the clusters on separate subplots (True), or all 
                    on the same plot (False)
        colour      (bool) to cycle through colours (True) or plot in black nd white (False)
        newfig      (bool) whether to make a new matplotlib figure(True) or use the current 
                    figure (False)
        in_a_row    (bool) if true, plot the subplots (if using) in a single row. Else make the 
                    subplots approximately square.
        joined      (bool) if true, connect the data points with lines
        gpplot      (bool) if true, plto the posterior of the GP for each cluster.
        min_in_cluster (float) ignore clusterse with less total assignemnt than this
        data_in_grey (bool) whether the data should be plotted in black and white.
        numbered    (bool) whether to include numbers on the top-right of each subplot.
        data_in_replicate (bool) whether to assume the data are in replicate, and plot the mean 
                          of each replicate instead of each data point
        fixed_inputs (list of tuples, as GPy.GP.plot) for GPs defined on more that one input, 
                     we'll plot a slice of the GP. this list defines how to fix the remaining 
                     inputs.
        ylim        (tuple) the limits to set on the y-axes.
        """

        #work out what input dimensions to plot
        fixed_dims = np.array([i for i,v in fixed_inputs])
        free_dims = np.setdiff1d(np.arange(X.shape[1]),fixed_dims)
        assert len(free_dims)==1, "can only plot mixtures of 1D functions"

        #figure, subplots
        if newfig:
            fig = plt.figure()
        else:
            fig = plt.gcf()

        if data_in_replicate:
            X_ = X[:, free_dims].flatten()
            X = np.unique(X_).reshape(-1,1)
            Y = np.vstack([Y[:,X_==x].mean(1) for x in X.flatten()]).T

        else:
            Y = Y
            X = X[:,free_dims]

        #find some sensible y-limits for the plotting
        if ylim is None:
            ymin,ymax = Y.min(), Y.max()
            ymin,ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
        else:
            ymin, ymax = ylim

        #work out how many clusters we're going to plot.
        Ntotal = np.sum(model.get_phihat() > min_in_cluster)
        if on_subplots:
            if in_a_row:
                Nx = 1
                Ny = Ntotal
            else:
                Nx = np.floor(np.sqrt(Ntotal))
                Ny = int(np.ceil(Ntotal/Nx))
                Nx = int(Nx)
        else:
            ax = plt.gca() # this seems to make new ax if needed

        #limits of GPs
        xmin,xmax = X.min(), X.max()
        xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)

        Xgrid = np.empty((300,X.shape[1]))
        Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
        for i,v in fixed_inputs:
            Xgrid[:,i] = v

        subplot_count = 0
        for i, ph, mu, var in zip(range(model.num_clusters), model.get_phihat(), *model.predict_components(Xgrid)):
            if ph>(min_in_cluster):
                ii = np.argmax(model.get_phi(),1)==i
                num_in_clust = np.sum(ii)
                if not np.any(ii):
                    continue
                if on_subplots:
                    ax = fig.add_subplot(Nx,Ny,subplot_count+1)
                    subplot_count += 1
                col='k'
                if joined:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker=None, linewidth=0.2,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker=None, linewidth=0.2,alpha=1)
                else:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker='.', linewidth=0.0,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker='.', linewidth=0.0,alpha=1)

                if numbered and on_subplots:
                    ax.text(1,1,str(int(num_in_clust)),transform=ax.transAxes,ha='right',va='top',bbox={'ec':'k','lw':1.3,'fc':'w'})

                err = 2*np.sqrt(np.diag(self.Lambda_inv[i,:,:]))
                if errorbars:ax.errorbar(self.X.flatten(), self.muk[:,i], yerr=err,ecolor=col, elinewidth=2, linewidth=0)

                ax.set_ylim(ymin, ymax)

        ax.set_xlim(xmin,xmax)
        plt.show()

def MOGplot(model,X):
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
