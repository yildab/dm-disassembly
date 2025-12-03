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
import scipy as sp

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

target_name = "Argon"

# input parameters

# %%
# element mass number
elements = np.array([16, 28, 27, 56, 40, 23, 39, 24, 48, 57, 59, 31, 32])

# percent by weight of each element
crust = np.array([46.7, 27.7, 8.1, 5.1, 3.7, 2.8, 2.6, 2.1, 0.6, 0.0, 0.0, 0.0, 0.0])/100
mantle = np.array([44.3, 21.3, 2.3, 6.3, 2.5, 0.0, 0.0, 22.3, 0.0, 0.2, 0.0, 0.0, 0.0])/100
core = np.array([0.0, 0.0, 0.0, 84.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.6, 0.3, 0.6, 9.0])/100

def go_to_boundary_new(posn, vhat, Rboundary):
    # write equation of sphere in at^2 + bt + c form

    a = vhat[0]**2 + vhat[1]**2 + vhat[2]**2
    b = 2*vhat[0]*posn[0] + 2*vhat[1]*posn[1]+ 2*vhat[2]*posn[2]
    c = posn[0]**2 + posn[1]**2 + posn[2]**2 - Rboundary**2

    disc = b**2 - 4*a*c

    # print(disc, a, b, c)
    if disc < 0:
        return posn
    else:
        ts = np.roots([a, b, c])
        #print(ts, ts[np.where(ts>0)])
        try:
            tval = np.min(ts[np.where(ts>=0)])
        except ValueError:
            return posn
        dx = vhat[0]*tval
        dy = vhat[1]*tval
        dz= vhat[2]*tval
        return [posn[0] + dx, posn[1] + dy, posn[2] + dz]


def composition(r): # r in km
    if r <= 3480*km:
        return core
    elif r <= 6346*km:
        return  mantle
    elif r <= 6371*km:
        return crust
    else:
        return np.zeros_like(crust)

def rho(r): # r in km, density in g/cm^3
    x = r / (6371*km) # renormalized radius from PREM

    if r <= 1221.5*km:
        return 13.0885 - 8.8381*x**2
    elif r <= 3480*km:
        return 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
    elif r <= 5701*km:
        return 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3
    elif r <= 5771*km:
        return 5.3197 - 1.4836*x
    elif r <= 5971*km:
        return 11.2494 -8.0298*x
    elif r <= 6151*km:
        return 7.1089 - 3.8045*x
    elif r <= 6346.6*km:
        return 2.6910 + 0.6924*x
    elif r <= 6356*km:
        return 2.9
    elif r <= 6368*km:
        return 2.6
    elif r <= 6371*km:
        return 1.02
    else:
        return 0
    
def n_composition(r):
    ns = [(rho(r)*composition(r)[val]*gram)/(elements[val]*mp) for val in range(len(elements))] # atoms per cm^3
    return ns

def nA(Aind, r):
    return n_composition(r)[Aind]


# %%
# MEAN FREE PATH

def mu(a, b):
    return a*b/(a+b)

def sigmaAd(Aind, sigmand, mx): # cm^2
    A = elements[Aind]
    mA = A*mp
    sigmaAd = A**2 * sigmand * (mu(mA, mx)/mu(mp, mx))**2# * helm2(q, Rnuc, sk) DO THIS PART LATER

    return sigmaAd

def lambdaMFP(r, sigmand, mx): # in km
    lambdaT = 0
    for Aind in range(len(elements)):
        lambdainv = nA(Aind, r)*sigmaAd(Aind, sigmand, mx) # cm
        if lambdainv != 0:
            lambdaT += lambdainv
    return 1/lambdaT * cm

def return_spread(data):
    mx = data[0]
    sigmand = data[1]

    print("== Run for mx = {} GeV, sigma = {} cm^2.".format(mx, sigmand))
    filesX = np.sort([str(f) for f in pathlib.Path().glob("DataOct/Xs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesY = np.sort([str(f) for f in pathlib.Path().glob("DataOct/Ys_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesZ = np.sort([str(f) for f in pathlib.Path().glob("DataOct/Zs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesV = np.sort([str(f) for f in pathlib.Path().glob("DataOct/Vs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesT = np.sort([str(f) for f in pathlib.Path().glob("DataOct/Ts_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    ind = filesX[0].find("angle-")+len("angle-")
    angles = [f[ind:ind+15] for f in filesX]

    summary_array = []
    if len(filesX) == 0:
        return 0


    for i in range(len(filesX)):

        xst = np.load(filesX[i], allow_pickle=True)
        yst = np.load(filesY[i], allow_pickle=True)
        zst = np.load(filesZ[i], allow_pickle=True)
        vst = np.load(filesV[i], allow_pickle=True)
        ts = np.load(filesT[i], allow_pickle=True)

        

        # take the final locations
        xs = xst[-1]
        ys = yst[-1]
        zs = zst[-1]
        vs = vst[-1]
        # rescale things so everything is exactly at earth surface, but only if the constituents are close enough 
        rs = np.sqrt(xs**2 + ys**2 + zs**2)

        discarded_points = np.where(rs <= 0.999*Rearth)[0]
        rescale_points = np.where(rs > 0.999*Rearth)[0]

        if len(discarded_points) > 0:
            print(discarded_points, rs[discarded_points], vs[discarded_points])

        vhatfs = np.stack(np.array([xst[-1] - xst[-5], yst[-1] - yst[-5], zst[-1] - zst[-5]]), axis=1)

        finalposns = np.array([go_to_boundary_new([xst[-1][i], yst[-1][i], zst[-1][i]], vhatfs[i], Rearth) for i in range(len(vhatfs))])
        xs = finalposns[:,0]
        ys = finalposns[:,1]
        zs = finalposns[:,2]

        rs_new = np.sqrt(xs**2 + ys**2 + zs**2)

        dist = rs_new - rs

        mfps = [lambdaMFP(r, sigmand, mx) for r in rs]

        ts_new = np.where(dist > mfps, np.nan, ts + dist/vs)
        ts = ts_new




        avg_v = np.mean(vs)

        refx = np.mean(xs)
        refz = np.mean(zs)

        refy = np.sqrt(Rearth**2 - refx**2 - refz**2)*np.sign(sp.stats.mode(ys)[0])


        angles_x = np.array(np.arccos((xs*refx+ys*refy)/(np.linalg.norm(np.stack([xs, ys], axis = 1), axis=1)*np.linalg.norm([refx, refy]))))



        angles_z = np.array(np.arccos((ys*refy+zs*refz)/(np.linalg.norm(np.stack([ys, zs], axis = 1), axis=1)*np.linalg.norm([refy, refz]))))

        angles_z = np.where(np.isnan(angles_z), 1e-15, angles_z)
        angles_x = np.where(np.isnan(angles_x), 1e-15, angles_x)

        z_arc_dist = Rearth*angles_z*np.sign(zs)
        x_arc_dist = Rearth*angles_x*np.sign(xs)

        avg_x = np.mean(x_arc_dist)
        avg_z = np.mean(z_arc_dist)


        spread_rs = np.sqrt((x_arc_dist - avg_x)**2 + (z_arc_dist - avg_z)**2)

        z_score = np.abs(sp.stats.zscore(spread_rs, nan_policy='propagate'))

        z_score = np.where(np.isnan(z_score), 0, z_score)

        nonoutlier_indices = np.where(z_score < 5)[0]

        try:
            center = np.min(np.where(spread_rs[nonoutlier_indices] == np.min(spread_rs[nonoutlier_indices])))

        except ValueError:
            center = 0
        
        tcenter = ts[nonoutlier_indices][center]

        tdiff = [np.abs(t - tcenter) for t in ts[nonoutlier_indices]]

        loc1, alpha1 = list(sp.stats.maxwell.fit(spread_rs[nonoutlier_indices], floc=0))    



        max_r = max(np.max(spread_rs)*4, 0.001)
        rrange = np.linspace(0, max_r, 1000)
        maxwell_cdf = sp.stats.maxwell.cdf(rrange, loc=loc1, scale=alpha1)

        spreadsize = np.min(rrange[np.where(maxwell_cdf > 0.9)])
        dtdr = (np.max(tdiff) - np.min(tdiff))/max(1000*spreadsize, 1)

        spread_points = np.where(rs <= 1.1*spreadsize)

        if len(xst) < 3:
            spreadsize = np.nan

        summary_array.append([mx, sigmand, angles[i], spreadsize, avg_v])

        pd.to_pickle(np.array(xs[nonoutlier_indices]), "DataOctProcessed/FinalXs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(ys[nonoutlier_indices]), "DataOctProcessed/FinalYs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(zs[nonoutlier_indices]), "DataOctProcessed/FinalZs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(vs[nonoutlier_indices]), "DataOctProcessed/FinalVs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(spread_rs[nonoutlier_indices]), "DataOctProcessed/SpreadRs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(ts[nonoutlier_indices]), "DataOctProcessed/FinalTs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))

    pd.to_pickle(np.array(summary_array), "DataOctProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand))

    print("All done saving!")

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
    
    doing_tasks = pool.map(return_spread, args)
    
    print("Processes are all done, in {} minutes!".format((time.time() - start_time)/60))
