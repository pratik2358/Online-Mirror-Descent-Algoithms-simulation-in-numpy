__author__ = "Pratik Karmakar"

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
import itertools
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy import stats
from matplotlib import cm

def hessian(fun):
    return jit(jacfwd(jacrev(fun)))
def gradient(fn,x):
    x=np.array(x,dtype="float64")
    if fn==l1:
        return grad(l1)(x)
    elif fn==l2:
        return grad(l2)(x)
    elif fn==l5:
        return grad(l5)(x)

def omd11(breg_fn,x1,cons,loss,lr,n,tol=1e-4,vlr=False):
    """
    OMD update with gradient descent and projection using Bregman divergence
    For linear box constraint
    
    breg_fn: A convex function used for bregman divergence
    x1: Starting point
    cons: Constraint (list input:[x1,x2,y1,y2]) (corner points of a box)
    loss: Surface to find minima on
    lr: learning rate
    n: Number of maximum iterations
    vlr: learning rate will be variable(decreasing) if set True, False by default
    """
    xi=np.array(x1,dtype="float32")
    loss_list=[]
    X=[xi]
    bounds=[(cons[0],cons[1]),(cons[2],cons[3])]
    y=np.array([0,0],dtype="float32")
    G=[]
    obj=[]
    steps=0
    obj_fun = lambda x: breg_fn(x)-breg_fn(y)-grad(breg_fn)(y)@(x-y)
    for i in range(10):
        loss_list.append(loss(xi))
        gi=grad(loss)(xi)
        G.append(np.linalg.norm(gi))
        if vlr:
            lr=lr/np.sqrt(i+1)
        xi-=lr*gi
        obj.append(obj_fun(xi))
        y=xi
        res = minimize(obj_fun, x0=xi, bounds=bounds)
        xi=res.x
        X.append(xi)
        steps+=1
    for i in range(n-10):
        if [abs(loss_list[len(loss_list)-k-1]-loss_list[len(loss_list)-k-2]) for k in range(len(loss_list)-1)]>[tol]:
            loss_list.append(loss(xi))
            gi=grad(loss)(xi)
            G.append(np.linalg.norm(gi))
            if vlr:
                lr=lr/np.sqrt(i+11)
            xi-=lr*gi
            obj.append(obj_fun(xi))
            y=xi
            res = minimize(obj_fun, x0=xi, bounds=bounds)
            xi=res.x
            X.append(xi)
            steps+=1
        else:
            break

    return {"points":X,"loss":loss,"loss_list":loss_list,"grad_norm":G,"obj":obj,"steps":steps}

def omd12(breg_fn,x1,cons,loss,eta0,n,tol=1e-4,vlr=False):
    """
    OMD using Bregman divergence as regularizer function
    For box linear constraint

    breg_fn: A convex function used for bregman divergence
    x1: Starting point
    cons: Constraint (list input:[x1,x2,y1,y2]) (corner points of a box)
    loss: Surface to find minima on
    eta0: Penalty for Bregman divergence
    n: Number of maximum iterations(>10)
    vlr: Penalty will be variable(decreasing) if set True, False by default
    """
    xi=np.array(x1,dtype="float32")
    loss_list=[]
    X=[xi]
    bounds=Bounds([cons[0],cons[2]],[cons[1],cons[3]])
    y=np.array([0.,0.],dtype="float32")
    gi=np.array([0.,0.],dtype="float32")
    G=[]
    obj=[]
    steps=0
    obj_fun = lambda x: (breg_fn(x)-breg_fn(y)-grad(breg_fn)(y)@(x-y))/eta0+gi@x
    for i in range(10):
        gi=grad(loss)(xi)
        obj.append(obj_fun(xi))
        loss_list.append(loss(xi))
        G.append(np.linalg.norm(grad(obj_fun)(xi)))
        if vlr:
            eta0/=np.sqrt(i+1)
        y=xi
        res = minimize(obj_fun, x0=xi,method='trust-constr', jac=grad(obj_fun),hess=jit(jacfwd(jacrev(obj_fun))), bounds=bounds,)
        xi=res.x
        X.append(xi)
        steps+=1
    for i in range(n-10):
        if [abs(loss_list[len(loss_list)-k-1]-loss_list[len(loss_list)-k-2]) for k in range(len(loss_list)-1)]>[tol]:
            gi=grad(loss)(xi)
            obj.append(obj_fun(xi))
            loss_list.append(loss(xi))
            G.append(np.linalg.norm(grad(obj_fun)(xi)))
            if vlr:
                eta0/=np.sqrt(i+11)
            y=xi
            res = minimize(obj_fun, x0=xi,method='trust-constr', jac=grad(obj_fun),hess=jit(jacfwd(jacrev(obj_fun))), bounds=bounds,)
            xi=res.x
            X.append(xi)
            steps+=1
        else:
            break
    return {"points":X,"loss":loss,"loss_list":loss_list,"grad_norm":G,"obj":obj,"steps":steps}

def omd21(breg_fn,x1,xc,yc,r,loss,lr,n,tol=1e-4,vlr=False):
    """
    OMD update with gradient descent and projection using Bregman divergence
    For circular constraint

    breg_fn: A convex function used for bregman divergence
    x1: Starting point
    xc: x coordinate of the centre of the constraint
    yc: y coordinate of the centre of the constraint
    r= Radius of the constraint
    loss: Surface to find minima on
    lr: learning rate
    n: Number of maximum iterations(>10)
    vlr: Penalty(eta0) will be variable(decreasing) if set True, False by default
    """
    xi=np.array(x1,dtype="float32")
    loss_list=[]
    X=[xi]
    cons = [{"type": "ineq", "fun": lambda x: -(x[0]-xc)**2-(x[1]-yc)**2+r**2}]
    y=np.array([0,0],dtype="float32")
    G=[]
    obj=[]
    obj_fun = lambda x: breg_fn(x)-breg_fn(y)-grad(breg_fn)(y)@(x-y)
    steps=0
    for i in range(10):
        loss_list.append(loss(xi))
        gi=grad(loss)(xi)
        obj.append(obj_fun(xi))
        xi-=lr*gi
        G.append(np.linalg.norm(gi))
        if vlr:
            lr/=np.sqrt(i+1)
        if (xi[0]-xc)**2+(xi[1]-yc)**2<=r**2:
            xi=xi
        else:
            y=xi
            res = minimize(obj_fun, x0=xi,constraints=cons)
            xi=res.x
        X.append(xi)
        steps+=1
    for i in range(n-10):
        if [abs(loss_list[len(loss_list)-k-1]-loss_list[len(loss_list)-k-2]) for k in range(len(loss_list)-1)]>[tol]:
            loss_list.append(loss(xi))
            gi=grad(loss)(xi)
            obj.append(obj_fun(xi))
            xi-=lr*gi
            G.append(np.linalg.norm(gi))
            if vlr:
                lr/=np.sqrt(i+11)
            if (xi[0]-xc)**2+(xi[1]-yc)**2<=r**2:
                xi=xi
            else:
                y=xi
                res = minimize(obj_fun, x0=xi,constraints=cons)
                xi=res.x
            X.append(xi)
            steps+=1
        else:
            break
    return {"points":X,"loss":loss,"loss_list":loss_list,"grad_norm":G,"obj":obj,"steps":steps}

def omd22(breg_fn,x1,xc,yc,r,loss,eta0,n,tol=1e-4,vlr=False):
    """
    OMD using Bregman divergence as regularizer function
    For circular constraint

    breg_fn: A convex function used for bregman divergence
    x1: Starting point
    xc: x coordinate of the centre of the constraint
    yc: y coordinate of the centre of the constraint
    r: Radius of the constraint
    loss: Surface to find minima on
    eta0: Penalty for bregman divergence term
    n: Number of maximum iterations(>10)
    vlr: Penalty(eta0) will be variable(decreasing) if set True, False by default
    """
    xi=np.array(x1,dtype="float32")
    loss_list=[]
    X=[xi]
    cons = [{"type": "ineq", "fun": lambda x: -(x[0]-xc)**2-(x[1]-yc)**2+r**2}]
    y=np.array([0,0],dtype="float32")
    gi=np.array([0,0],dtype="float32")
    G=[]
    obj=[]
    obj_fun = lambda x: (breg_fn(x)-breg_fn(y)-grad(breg_fn)(y)@(x-y))/eta0+gi@x
    steps=0
    for i in range(10):
        loss_list.append(loss(xi))
        gi=grad(loss)(xi)
        G.append(np.linalg.norm(gi))
        obj.append(obj_fun(xi))
        if vlr:
            eta0/=np.sqrt(i+1)
        y=xi
        res = minimize(obj_fun, x0=xi,method='trust-constr', jac=grad(obj_fun),hess=jit(jacfwd(jacrev(obj_fun))),constraints=cons)
        xi=res.x
        X.append(xi)
        steps+=1
    for i in range(n-10):
        if [abs(loss_list[len(loss_list)-k-1]-loss_list[len(loss_list)-k-2]) for k in range(len(loss_list)-1)]>[tol]:
            loss_list.append(loss(xi))
            gi=grad(loss)(xi)
            G.append(np.linalg.norm(gi))
            obj.append(obj_fun(xi))
            if vlr:
                eta0/=np.sqrt(i+11)
            y=xi
            res = minimize(obj_fun, x0=xi,method='trust-constr', jac=grad(obj_fun),hess=jit(jacfwd(jacrev(obj_fun))),constraints=cons)
            xi=res.x
            X.append(xi)
            steps+=1
        else:
            break
    return {"points":X,"loss":loss,"loss_list":loss_list,"grad_norm":G,"obj":obj,"steps":steps}

def V_set(x):
    """
    Random projections on probability simplex used in OMD with expert advice algorithms
    """
    M=np.array([np.exp(x)/sum(np.exp(x)),x**4/np.sum(x**4),x**6/np.sum(x**6),abs(x)/np.sum(x),abs(np.sin(x))/np.sum(abs(np.sin(x)))])
    return M

def exp_adv1(x1,f,n,eta0):
    """
    x1: Initial probability distribution
    f: Loss function
    n: Number of maximum iterations
    eta0: Learning rate
    """
    exps=np.eye(len(x1))
    xk=np.arange(len(x1))
    p=x1
    loss_list=[]
    regret=[]
    P=[p]
    grad_list=[]
    reg=0
    for k in range(n):
        dist=stats.rv_discrete(name='dist',values=(xk,p))
        i=dist.rvs(size=1)
        expert=exps[int(i)]
        loss=f(p)
        gi=grad(f)(p)
        grad_list.append(gi)
        reg=sum([g@p for g,p in zip(grad_list,P)])-sum([g@expert for g in grad_list])
        regret.append(reg)
        loss_list.append(loss)
        p=p*np.exp(-eta0*gi)/sum(p*np.exp(-eta0*gi))
        P.append(p)
        print(p)
    return{"loss_list":loss_list,"points":P,"regret":regret}

def exp_adv2(x1,f,n,eta0):
    """
    x1: Initial probability distribution
    f: Loss function
    n: Number of maximum iterations
    eta0: Learning rate
    """
    p=x1
    loss_list=[]
    regret_list=[]
    P=[p]
    grad_list=[]
    reg=0
    for k in range(n):
        a=np.random.randn(*x1.shape)
        exps=V_set(a)
        i=np.random.randint(len(exps))
        expert=exps[i]
        loss=f(p)
        gi=grad(f)(p)
        grad_list.append(gi)
        reg=sum([g@p for g,p in zip(grad_list,P)])-sum([g@expert for g in grad_list])
        regret_list.append(reg)
        loss_list.append(loss)
        p=p*np.exp(-eta0*gi)/sum(p*np.exp(-eta0*gi))
        P.append(p)
        print(p)
    return{"loss_list":loss_list,"points":P,"regret":regret_list}

def exp_adv3(x1,f,n,eta0,eps=0.05):
    """
    x1: Initial probability distribution
    f: Loss function
    n: Number of maximum iterations
    eta0: Learning rate
    eps: Amount of perturbation to add to p (decides amount of randomization in expert)
    """
    xk=np.arange(len(x1))
    p=x1
    loss_list=[]
    regret_list=[]
    P=[p]
    grad_list=[]
    reg=0
    for k in range(n):
        a=p+eps*np.random.randn(*x1.shape)
        exps=V_set(a)
        dist=stats.rv_discrete(name='dist',values=(xk,p))
        i=int(dist.rvs(size=1))%5
        expert=exps[i]
        loss=f(p)
        gi=grad(f)(p)
        grad_list.append(gi)
        reg=sum([g@p for g,p in zip(grad_list,P)])-sum([g@expert for g in grad_list])
        regret_list.append(reg)
        loss_list.append(loss)
        p=p*np.exp(-eta0*gi)/sum(p*np.exp(-eta0*gi))
        P.append(p)
        print(p)
    return{"loss_list":loss_list,"points":P,"regret":regret_list}

def exp_adv4(x1,breg_fn,f,n,eta0):
    """
    Expert advice using Bregman divergence as regularizer
    x1: Initial probability distribution
    f: Loss function
    n: Number of maximum iterations
    eta0: Learning rate
    """
    exps=np.eye(len(x1))
    xk=np.arange(len(x1))
    p=x1
    cons = [{"type": "eq", "fun": lambda x: np.sum(x)-1}]
    loss_list=[]
    regret=[]
    P=[p]
    grad_list=[]
    reg=0
    y=np.random.randn(len(p))
    y=np.exp(y)/sum(np.exp(y))
    gi=np.zeros(len(p),dtype="float32")
    obj=[]
    obj_fun = lambda x: (breg_fn(x)-breg_fn(y)-grad(breg_fn)(y)@(x-y))/eta0+gi@x
    for k in range(n):
        dist=stats.rv_discrete(name='dist',values=(xk,p))
        i=dist.rvs(size=1)
        expert=exps[int(i)]
        loss=f(p)
        gi=grad(f)(p)
        obj.append(obj_fun(p))
        grad_list.append(gi)
        reg=sum([g@p for g,p in zip(grad_list,P)])-sum([g@expert for g in grad_list])
        regret.append(reg)
        loss_list.append(loss)
        y=p
        opt = minimize(obj_fun, x0=p,method='trust-constr', jac=grad(obj_fun),hess=jit(jacfwd(jacrev(obj_fun))),constraints=cons, bounds=[(0, 1)] * len(p))
        p=opt.x
        P.append(p)
        print(p)
    return{"loss_list":loss_list,"points":P,"regret":regret}



def comp_plot1(breg_fn,omd,x1,cons,loss,lr,n,tol=1e-4,vlr=False,figsize=(60,15)):
    """
    Comparison plots for linear constraint

    breg_fn: list of functions
    omd: omd function
    x1: list of starting points
    cons: list of constraints (list of lists)
    loss: list of loss functions
    lr: list of learning rates
    n: maximum iterations
    tol: tolerance of change
    vlr: valriable learning rate if set True
    """
    if len(breg_fn)>1:
        for i in range(len(breg_fn)):
            res=omd(breg_fn[i],x1[0],cons[0],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-3,cons[0][1]+3,100)
            y=np.linspace(cons[0][2]-3,cons[0][3]+3,100)
            X,Y=np.meshgrid(x,y)


            z=loss[0]((X,Y))

            resx=[a for (a,b) in res["points"]]
            resy=[b for (a,b) in res["points"]]

            plt.figure(figsize=figsize)
            plt.subplot(2,len(breg_fn),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(resx,resy,color="red")
            plt.plot(resx,resy,color="red")
            plt.plot([cons[0][0],cons[0][1],cons[0][1],cons[0][0],cons[0][0]],[cons[0][2],cons[0][2],cons[0][3],cons[0][3],cons[0][2]],color="black")
            plt.scatter(resx[-1],resy[-1],marker="X",color="black",s=100)
            plt.title("Bregman function="+str(breg_fn[i])+"\n"+"steps="+str(res["steps"]))

    elif len(x1)>1:
        for i in range(len(x1)):
            res=omd(breg_fn[0],x1[i],cons[0],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-3,cons[0][1]+3,100)
            y=np.linspace(cons[0][2]-3,cons[0][3]+3,100)
            X,Y=np.meshgrid(x,y)


            z=loss[0]((X,Y))

            resx=[a for (a,b) in res["points"]]
            resy=[b for (a,b) in res["points"]]

            plt.figure(figsize=figsize)
            plt.subplot(2,len(x1),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(resx,resy,color="red")
            plt.plot(resx,resy,color="red")
            plt.plot([cons[0][0],cons[0][1],cons[0][1],cons[0][0],cons[0][0]],[cons[0][2],cons[0][2],cons[0][3],cons[0][3],cons[0][2]],color="black")
            plt.scatter(resx[-1],resy[-1],marker="X",color="black",s=100)
            plt.title("Starting point="+str(x1[i])+"\n"+"steps="+str(res["steps"]))

    elif len(cons)>1:
        for i in range(len(cons)):
            res=omd(breg_fn[0],x1[0],cons[i],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[i][0]-3,cons[i][1]+3,100)
            y=np.linspace(cons[i][2]-3,cons[i][3]+3,100)
            X,Y=np.meshgrid(x,y)


            z=loss[0]((X,Y))

            resx=[a for (a,b) in res["points"]]
            resy=[b for (a,b) in res["points"]]

            plt.figure(figsize=figsize)
            plt.subplot(2,len(cons),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(resx,resy,color="red")
            plt.plot(resx,resy,color="red")
            plt.plot([cons[i][0],cons[i][1],cons[i][1],cons[i][0],cons[i][0]],[cons[i][2],cons[i][2],cons[i][3],cons[i][3],cons[i][2]],color="black")
            plt.scatter(resx[-1],resy[-1],marker="X",color="black",s=100)
            plt.title("Constraint="+str(cons[i])+"\n"+"steps="+str(res["steps"]))

    elif len(loss)>1:
        for i in range(len(loss)):
            res=omd(breg_fn[0],x1[0],cons[0],loss[i],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-3,cons[0][1]+3,100)
            y=np.linspace(cons[0][2]-3,cons[0][3]+3,100)
            X,Y=np.meshgrid(x,y)


            z=loss[i]((X,Y))

            resx=[a for (a,b) in res["points"]]
            resy=[b for (a,b) in res["points"]]

            plt.figure(figsize=figsize)
            plt.subplot(2,len(loss),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(resx,resy,color="red")
            plt.plot(resx,resy,color="red")
            plt.plot([cons[0][0],cons[0][1],cons[0][1],cons[0][0],cons[0][0]],[cons[0][2],cons[0][2],cons[0][3],cons[0][3],cons[0][2]],color="black")
            plt.scatter(resx[-1],resy[-1],marker="X",color="black",s=100)
            plt.title("Loss function="+str(loss[i])+"\n"+"steps="+str(res["steps"]))

    elif len(lr)>1:
        for i in range(len(lr)):
            res=omd(breg_fn[0],x1[0],cons[0],loss[0],lr[i],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-3,cons[0][1]+3,100)
            y=np.linspace(cons[0][2]-3,cons[0][3]+3,100)
            X,Y=np.meshgrid(x,y)


            z=loss[0]((X,Y))

            resx=[a for (a,b) in res["points"]]
            resy=[b for (a,b) in res["points"]]

            plt.figure(figsize=figsize)
            plt.subplot(2,len(lr)//2,i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(resx,resy,color="red")
            plt.plot(resx,resy,color="red")
            plt.plot([cons[0][0],cons[0][1],cons[0][1],cons[0][0],cons[0][0]],[cons[0][2],cons[0][2],cons[0][3],cons[0][3],cons[0][2]],color="black")
            plt.scatter(resx[-1],resy[-1],marker="X",color="black",s=100)
            if omd==omd12:
                plt.title("Penalty="+str(lr[i])+"\n"+"steps="+str(res["steps"]))
            elif omd==omd11:
                plt.title("Learning rate="+str(lr[i])+"\n"+"steps="+str(res["steps"]))

def comp_plot2(breg_fn,omd,x1,cons,loss,lr,n,tol=1e-4,vlr=False,figsize=(60,15)):
    """
    Comparison plots for non-linear constraint

    breg_fn: list of functions
    omd: omd function
    x1: list of starting points
    cons: list of constraints (list of lists)
    loss: list of loss functions
    lr: list of learning rates
    n: maximum iterations
    tol: tolerance of change
    vlr: valriable learning rate if set True
    """
    if len(breg_fn)>1:
        for i in range(len(breg_fn)):
            res2=omd(breg_fn[i],x1[0],cons[0][0],cons[0][1],cons[0][2],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-cons[0][2]-3,cons[0][0]+cons[0][2]+3,100)
            y=np.linspace(cons[0][1]-cons[0][2]-3,cons[0][1]+cons[0][2]+3,100)
            X,Y=np.meshgrid(x,y)

            z=loss[0]((X,Y))

            res2x=[a for (a,b) in res2["points"]]
            res2y=[b for (a,b) in res2["points"]]

            theta=np.linspace(0,2*np.pi,100)
            cx=cons[0][0]+cons[0][2]*np.cos(theta)
            cy=cons[0][1]+cons[0][2]*np.sin(theta)

            plt.figure(figsize=figsize)
            plt.subplot(2,len(breg_fn),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(res2x,res2y,color="red")
            plt.plot(res2x,res2y,color="red")
            plt.plot(cx,cy)
            plt.Circle((cons[0][0],cons[0][1]),cons[0][2])
            plt.scatter(res2x[-1],res2y[-1],marker="X",color="black",s=100)
            plt.title("Bregman function="+str(breg_fn[i])+"\n"+"steps="+str(res2["steps"]))
    elif len(x1)>1:
        for i in range(len(x1)):
            res2=omd(breg_fn[0],x1[i],cons[0][0],cons[0][1],cons[0][2],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-cons[0][2]-3,cons[0][0]+cons[0][2]+3,100)
            y=np.linspace(cons[0][1]-cons[0][2]-3,cons[0][1]+cons[0][2]+3,100)
            X,Y=np.meshgrid(x,y)

            z=loss[0]((X,Y))

            res2x=[a for (a,b) in res2["points"]]
            res2y=[b for (a,b) in res2["points"]]

            theta=np.linspace(0,2*np.pi,100)
            cx=cons[0][0]+cons[0][2]*np.cos(theta)
            cy=cons[0][1]+cons[0][2]*np.sin(theta)

            plt.figure(figsize=figsize)
            plt.subplot(2,len(x1),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(res2x,res2y,color="red")
            plt.plot(res2x,res2y,color="red")
            plt.plot(cx,cy)
            plt.Circle((cons[0][0],cons[0][1]),cons[0][2])
            plt.scatter(res2x[-1],res2y[-1],marker="X",color="black",s=100)
            plt.title("Starting point="+str(x1[i])+"\n"+"steps="+str(res2["steps"]))
    elif len(cons)>1:
        for i in range(len(cons)):
            res2=omd(breg_fn[0],x1[0],cons[i][0],cons[i][1],cons[i][2],loss[0],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[i][0]-cons[i][2]-3,cons[i][0]+cons[i][2]+3,100)
            y=np.linspace(cons[i][1]-cons[i][2]-3,cons[i][1]+cons[i][2]+3,100)
            X,Y=np.meshgrid(x,y)

            z=loss[0]((X,Y))

            res2x=[a for (a,b) in res2["points"]]
            res2y=[b for (a,b) in res2["points"]]

            theta=np.linspace(0,2*np.pi,100)
            cx=cons[i][0]+cons[i][2]*np.cos(theta)
            cy=cons[i][1]+cons[i][2]*np.sin(theta)

            plt.figure(figsize=figsize)
            plt.subplot(2,len(cons),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(res2x,res2y,color="red")
            plt.plot(res2x,res2y,color="red")
            plt.plot(cx,cy)
            plt.Circle((cons[i][0],cons[i][1]),cons[i][2])
            plt.scatter(res2x[-1],res2y[-1],marker="X",color="black",s=100)
            plt.title("Constraint="+str(cons[i])+"\n"+"steps="+str(res2["steps"]))
    elif len(loss)>1:
        for i in range(len(loss)):
            res2=omd(breg_fn[0],x1[0],cons[0][0],cons[0][1],cons[0][2],loss[i],lr[0],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-cons[0][2]-3,cons[0][0]+cons[0][2]+3,100)
            y=np.linspace(cons[0][1]-cons[0][2]-3,cons[0][1]+cons[0][2]+3,100)
            X,Y=np.meshgrid(x,y)

            z=loss[i]((X,Y))

            res2x=[a for (a,b) in res2["points"]]
            res2y=[b for (a,b) in res2["points"]]

            theta=np.linspace(0,2*np.pi,100)
            cx=cons[0][0]+cons[0][2]*np.cos(theta)
            cy=cons[0][1]+cons[0][2]*np.sin(theta)

            plt.figure(figsize=figsize)
            plt.subplot(2,len(loss),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(res2x,res2y,color="red")
            plt.plot(res2x,res2y,color="red")
            plt.plot(cx,cy)
            plt.Circle((cons[0][0],cons[0][1]),cons[0][2])
            plt.scatter(res2x[-1],res2y[-1],marker="X",color="black",s=100)
            plt.title("Loss function="+str(loss[i])+"\n"+"steps="+str(res2["steps"]))
    elif len(lr)>1:
        for i in range(len(lr)):
            res2=omd(breg_fn[0],x1[0],cons[0][0],cons[0][1],cons[0][2],loss[0],lr[i],n,tol=tol,vlr=vlr)
            x=np.linspace(cons[0][0]-cons[0][2]-3,cons[0][0]+cons[0][2]+3,100)
            y=np.linspace(cons[0][1]-cons[0][2]-3,cons[0][1]+cons[0][2]+3,100)
            X,Y=np.meshgrid(x,y)

            z=loss[0]((X,Y))

            res2x=[a for (a,b) in res2["points"]]
            res2y=[b for (a,b) in res2["points"]]

            theta=np.linspace(0,2*np.pi,100)
            cx=cons[0][0]+cons[0][2]*np.cos(theta)
            cy=cons[0][1]+cons[0][2]*np.sin(theta)

            plt.figure(figsize=figsize)
            plt.subplot(2,len(lr),i+1)
            plt.contour(X,Y,z,100)
            plt.scatter(res2x,res2y,color="red")
            plt.plot(res2x,res2y,color="red")
            plt.plot(cx,cy)
            plt.Circle((cons[0][0],cons[0][1]),cons[0][2])
            plt.scatter(res2x[-1],res2y[-1],marker="X",color="black",s=100)
            if omd==omd22:
                plt.title("Penalty="+str(lr[i])+"\n"+"steps="+str(res2["steps"]))
            elif omd==omd21:
                plt.title("Learning rate="+str(lr[i])+"\n"+"steps="+str(res2["steps"]))
