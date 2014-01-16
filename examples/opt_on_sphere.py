import numpy as np
import matplotlib.pyplot as pb
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import sys
pb.ion()

def spherical_to_cart(theta,psi,r):
    """accept arrays of sperical coords (theta, psi, r) and return arrays of cartesian coords (x,y,z)"""
    x = np.cos(psi)*np.sin(theta)*r
    y = np.sin(psi)*np.sin(theta)*r
    z = np.cos(theta)*r
    return x,y,z
def cart_to_spherical(x,y,z):
    r =  np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    psi = np.arctan(y/x)
    return theta, psi, r

#plot a straight line in (psi,theta) on the sphere
tt, pp = np.linspace(0,2,100), np.linspace(0,2,100)
fig_flat = pb.figure()
pb.plot(tt,pp,linewidth=2)
fig_s = pb.figure()
ax = fig_s.add_subplot(111, projection='3d')
ax.plot(*spherical_to_cart(tt,pp,1.),linewidth=2)

#mesh the shpere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
xmesh = np.outer(np.cos(u), np.sin(v))
ymesh = np.outer(np.sin(u), np.sin(v))
zmesh = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(xmesh, ymesh, zmesh,  rstride=6, cstride=6, color='k')

#objective funcion: quadratic in x,y,z
def objective(x,y,z):
    x0,y0,z0 = spherical_to_cart(2,2,1)
    return (x0-x)**2 + (y0-y)**2 + (z0-z)**2
def grad(x,y,z):
    x0,y0,z0 = spherical_to_cart(2,2,1)
    return 2.*(x-x0),2.*(y-y0),2.*(z-z0)

#illustrate the objective
fig_s = pb.figure()
ax = fig_s.add_subplot(111, projection='3d')
f =-objective(xmesh,ymesh,zmesh)
f -= f.min()
f /= f.max()
ax.plot_surface(xmesh,ymesh,zmesh,rstride=4,cstride=4,facecolors=pb.cm.jet(f),alpha=1)

#optimisation in theta, psi
xpath = [0.001*np.random.randn(2)]
f_spherical = lambda x: objective(*spherical_to_cart(x[0],x[1],1))
def g_spherical(x):
    g_cart = grad(*spherical_to_cart(x[0],x[1],1))
    theta, psi = x
    J = np.array([[np.cos(psi)*np.cos(theta),np.sin(psi)*np.cos(theta),-np.sin(theta)],
        [-np.sin(psi)*np.sin(theta),np.cos(psi)*np.sin(theta),0.]])
    return np.dot(J,g_cart)
iteration=0
while True:
    search_dir = -g_spherical(xpath[-1])
    grad_norm = np.sum(np.square(search_dir))
    if grad_norm<1e-6:
        break
    alpha, fc,gc,foo,bar,baz = optimize.line_search(f_spherical,g_spherical,xpath[-1],search_dir,-search_dir)
    xnew = xpath[-1] + 0.01*alpha*search_dir
    xpath.append(xnew)
    iteration += 1
    print iteration,grad_norm,'\r',
    sys.stdout.flush()
print ''

xx,yy,zz = np.vstack(map(lambda x: spherical_to_cart(x[0],x[1],1),xpath)).T
ax.plot(xx,yy,zz,'mo',linewidth=2,mew=0)

#natural optimisation in theta, psi
xpath = [0.001*np.random.randn(2)]
iteration=0
while True:
    g = g_spherical(xpath[-1])
    #metric_inv = np.array([[1,0],[0,1]])
    #metric_inv = np.array([[1./(np.cos(xpath[-1][1])**2),0],[0,1]])
    metric_inv = np.array([[1,0],[0,1./(np.sin(xpath[-1][0])**2)]])
    natGrad = np.dot(metric_inv, g)
    grad_norm = np.sum(np.square(g))
    if (grad_norm<1e-6) or iteration >1e3:
        break
    search_dir = -natGrad
    alpha, fc,gc,foo,bar,baz = optimize.line_search(f_spherical,g_spherical,xpath[-1],search_dir,g)
    xnew = xpath[-1] + 0.01*alpha*search_dir
    xpath.append(xnew)
    iteration += 1
    print iteration,grad_norm,'\r',
    sys.stdout.flush()

xx,yy,zz = np.vstack(map(lambda x: spherical_to_cart(x[0],x[1],1),xpath)).T
ax.plot(xx,yy,zz,'co',linewidth=2,mew=0)











