# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
colors = matplotlib.colormaps['Dark2'].colors
plt.rcParams["figure.dpi"] = 600
import seaborn as sns
import pandas as pd
import itertools
import time
import multiprocessing
import pathlib
import scipy
import kdetools
from scipy.integrate import quad
# %%

# UNITS

mp = 0.938 # GeV
GeV = 1
gram = 5.62e23*GeV
cm = 1/(1000*100)
km = 1

Rearth = 6371*km
Rcore = 3480*km
Rmantle = 6346*km

sec=1
c = 300000*km/sec

target_name = "Xenon"

# input parameters

def mu(a, b):
    return a*b/(a+b)

def get_spreadsize(mx, sigmand):
    data = np.load("DataOctProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand), allow_pickle=True)

    spreadsizes = np.float64(data[:,3])

    return np.mean(spreadsizes)

def nAdetector(element):
    nAr = 1.4 * 6.022e23/40 # target atoms per cm^3
    nXe = 3.52 * 6.022e23/131 # target atoms per cm^3
    if element == "Xenon":
        return  nXe
    elif element == "Argon":
        return nAr
    return 0

def num_composites_earthpersec(Nd, mx):
    rhodm = 0.3*GeV/cm**3 # in m/km^3
    Md = Nd*mx*GeV
    veldm = 1e-3*c # km/s
    flux = rhodm*veldm/Md # km^(-2)s^-1
    return flux*(np.pi*Rearth**2)

def Rnuc(A):
    sk = 0.9/0.197 #(* nuclear skin thickness, GeV^-1 *)
    a =  0.52/0.197 #(* GeV^-1 *)
    R0 = (1.23*A**(1/3) - 0.6)/0.197 # GeV^-1
    Rnuc = (np.sqrt(R0**2 + (7/3)* np.pi**2* a**2 - 5*sk**2))

    return Rnuc

def helm2(q, r, s): #(* Helm form factor, squared, for momentum transfer q, radius r, and nuclear skin thickness s *)
    #q = np.sqrt(2*mA*Er)

    #if q == 0:
    #    return 1

    def j1(x):
        return np.sin(x)/x**2 - np.cos(x)/x; # spherical Bessel function
    Fa2 = (3*(j1(q*r)/(q*r)))**2*np.exp(-(q*s)**2)*np.heaviside(q*r-0.0001, 1)+np.heaviside(0.0001 - q*r, 1)

    return  Fa2


def get_velocity(mx, sigmand):
    data = np.load("DataOctProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand), allow_pickle=True)
    vs = np.float64(data[:,-1])
    return np.mean(vs)


def sigmaAddetector2(element, sigmand, mx, vel): # cm^2
    if element == "Xenon":
        A = 131
    elif element == "Argon":
        A = 40
    mA = A*mp

    Ermax = 2*mu(mx, mA)**2*(vel)**2/mA

    func_dsigmaAd = lambda Er: A**2 * sigmand * (mu(mA, mx)/mu(mp, mx))**2 * helm2(np.sqrt(2*mA*Er), Rnuc(A), s=0.9/0.197)* helm2(np.sqrt(2*mA*Er), Rnuc(A), s=0.9/0.197)*np.heaviside(np.sqrt(2)*vel*mu(mx, mA)- np.sqrt(2*mA*Er), 1)

    sigmaAd = mA/(2*(mu(mA, mx)*(vel))**2)*quad(func_dsigmaAd, 0, Ermax, epsabs=1.e-20, epsrel=1.e-20, limit=50)[0]#quad(func_dsigmaAd, 0, Ermax, epsabs=1.e-20, epsrel=1.e-20, limit=50)[0]

    return sigmaAd


def ScattersPerCone(mx, sigma, Nd, Rconedata, target='Xenon'):
    Rcone = Rconedata * 1000 * 100 # in cm
    Adetector = 1*(100**2) # cm^2
    l = 1 * 100 # cm

    constituentsindetector = min(Nd*Adetector/(np.pi*Rcone**2), Nd)
    vel = get_velocity(10**mx, 10**sigma)/300000
    scatterperconstituent = nAdetector(target)*sigmaAddetector2(target, 10**sigma, 10**mx, vel)*l
    return constituentsindetector*scatterperconstituent





def get_dt(data):
    # given a number of scatters per cone, what is dt between two scatters?
    mx = data[0]
    sigmand = data[1]
    Nds = [1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
    dts = []
    processes = []

    for Nd in Nds:

        # first find number of scatters, based on 
        rcone = get_spreadsize(mx, sigmand)
        num_scatters = ScattersPerCone(np.log10(mx), np.log10(sigmand), Nd, rcone)
        if num_scatters <= 1:
            dts.append(np.nan)
            processes.append(np.nan)

            continue

        print("== Run for mx = {} GeV, sigma = {} cm^2.".format(mx, sigmand))
        filesR = np.sort([str(f) for f in pathlib.Path().glob("DataOctProcessed/SpreadRs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        filesT = np.sort([str(f) for f in pathlib.Path().glob("DataOctProcessed/FinalTs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        # filesX = np.sort([str(f) for f in pathlib.Path().glob("DataMayFresh/FinalXs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        # filesY = np.sort([str(f) for f in pathlib.Path().glob("DataMayFresh/FinalYs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        # filesZ = np.sort([str(f) for f in pathlib.Path().glob("DataMayFresh/FinalZs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        filesV = np.sort([str(f) for f in pathlib.Path().glob("DataOctProcessed/FinalVs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        ind = filesR[0].find("angle-")+len("angle-")
        angles = [f[ind:ind+15] for f in filesR]
        proc = 0

        mean_tdiff = 0
        numangles = len(filesR)

        for i in range(numangles):

            scatter_ts = []

            rs = np.load(filesR[i], allow_pickle=True)
            ts = np.load(filesT[i], allow_pickle=True)#*1e9
            vs = np.load(filesV[i], allow_pickle=True)

            tsnew = ts[np.where(np.isnan(ts) == False)]
            rs = rs[np.where(np.isnan(ts) == False)]
            ts = tsnew



            z_score = np.abs(scipy.stats.zscore(np.log10(ts), nan_policy='propagate'))
            z_score = np.where(np.isnan(z_score), 0, z_score)

            nonoutliers = np.where(z_score < 2)[0]

            ts = ts[nonoutliers]
            rs = rs[nonoutliers]


            if np.max(ts) - np.min(ts) <= 1e-9:
                print("NO POINT IN RUNNING", mx, sigmand, angles[i])
                mean_tdiff +=(np.max(ts) - np.min(ts))/num_scatters
                proc += 0
            else:
                numtrials=100
                data = np.stack((rs, ts), axis=1)
                dtoverr = 0

                try:
                    time_kde = kdetools.gaussian_kde(data.T)

                    for j in range(numtrials):
                        rval = rcone*np.sqrt(np.random.rand())
                        tsample = time_kde.conditional_resample(1000, x_cond=np.array([rval]), dims_cond=[0]).ravel()
                        (mu, sigma) = scipy.stats.norm.fit(tsample)
                        twindow = 2*sigma
                        dt = twindow/num_scatters
                        dtoverr += dt/numtrials
                    print("NO EXCEPTION", mx, sigmand, angles[i], np.max(ts), np.min(ts), dtoverr)
                    proc += 1
                except:
                    dtoverr = 0
                    (mu, sigma) = scipy.stats.norm.fit(ts)
                    twindow = 2*sigma 
                    print("EXCEPTION:", mx, sigmand, angles[i], np.max(ts), np.min(ts), twindow)
                    dtoverr = twindow/num_scatters
                    proc += 2

                mean_tdiff += dtoverr
        dts.append(mean_tdiff/numangles)
        processes.append(proc)


    pd.to_pickle(np.array([Nds, dts, processes]), "DataOctProcessed/{}Nonoutliers/FinalDTs_mx-{}_sigma-{}.pkl".format(target_name, mx, sigmand))
    print("Done saving! -- ", dts)

    return 0




# DEFINING RunParameters
cpus = int(multiprocessing.cpu_count())
print("Working across {} CPUs.".format(cpus))
mxs = np.append(10**np.linspace(2, 10, 20), [10**np.float64(10.421052631578947), 10**np.float64(10.842105263157894), 10**np.float64(11.263157894736842), 10**np.float64(11.68421052631579), 10**np.float64(12.105263157894736)])
sigmas = 10**np.linspace(-41, -37, 20)

args = []

for i in range(len(mxs)):
    for j in range(len(sigmas)):
        args.append([mxs[i], sigmas[j]])

start_time = time.time()

if __name__ == "__main__":
    
    # SET UP MULTIPROCESSING

    num_pool = cpus

    pool = multiprocessing.Pool(processes = num_pool, maxtasksperchild=2)
    
    doing_tasks = pool.map(get_dt, args)
    
    print("Processes are all done, in {} minutes!".format((time.time() - start_time)/60))

# %%
