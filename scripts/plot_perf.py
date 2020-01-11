import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

parser = argparse.ArgumentParser()
 
parser.add_argument(
        "-data",
        type=str,
        default = "shared_memory.csv",
    )

args = parser.parse_args();

fn_data = args.data;

data = np.loadtxt(fn_data, delimiter=',',skiprows=1);

N = data[:,0];
gtx_titanx = data[:,1];
e5_2670_32 = data[:,2];
e5_2670_16 = data[:,3];
e5_2670_8 = data[:,4];
e5_2670_4 = data[:,5];
e5_2670_2 = data[:,6];
e5_2670_1 = data[:,7];

plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')


fig1 = plt.figure(1, figsize=(8,8), dpi=100);
plt.grid(axis='both',which='both');
plt.loglog(N[1::],gtx_titanx[1::],'o-',linewidth=2.5)
plt.loglog(N,e5_2670_32,'s-',linewidth=2.5)
plt.loglog(N,e5_2670_16,'s-',linewidth=2.5)
plt.loglog(N,e5_2670_8,'s-',linewidth=2.5)
plt.loglog(N,e5_2670_4,'s-',linewidth=2.5)
plt.loglog(N,e5_2670_2,'s-',linewidth=2.5)
plt.loglog(N,e5_2670_1,'s-',linewidth=2.5)

plt.ylabel('MLUPs', fontsize=18);
plt.xlabel('N',fontsize=18);
                

#plt.legend(['GTX Titan X', 'n=32','n=16','n=8','n=4','n=2','n=1'],fontsize=18,loc='right', ncol=1)
plt.show()

#fig1.savefig(str(NY) + "x" + str(NX) + "_u.png",bbox_inches='tight');

