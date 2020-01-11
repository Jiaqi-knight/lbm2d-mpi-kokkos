import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "-re",
        nargs="*",
        type=int,
    )

parser.add_argument(
        "-nx",
        type=int,
    )
 
parser.add_argument(
        "-ny",
        type=int,
    )
 
parser.add_argument(
        "-ulid",
        type=float,
    )
 
parser.add_argument(
        "-cuda",
        type=int,
        default=0,
    )
 
parser.add_argument(
        "-data",
        type=str,
        default = "ghia_data.csv",
    )

args = parser.parse_args();
 
reynolds = args.re;
NX = args.nx;
NY = args.ny;
ulid = args.ulid;
cuda = args.cuda;
fn_data = args.data;

data = np.loadtxt(fn_data, delimiter=',',skiprows=1);

plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

fig2 = plt.figure(2,figsize=(8,8), dpi=100);
plt.grid(axis='both',which='both');
plt.xlabel('x-position', fontsize=18);
plt.ylabel('v-component',fontsize=18);
plt.xlim([0, 1]);

fig1 = plt.figure(1, figsize=(8,8), dpi=100);
plt.grid(axis='both',which='both');
plt.ylabel('y-position', fontsize=18);
plt.xlabel('u-component',fontsize=18);
plt.ylim([0, 1]);

for re in reynolds:
    
    dir = "re" + str(re) + "_" + str(NY) + "x" + str(NY) + "/output"
    files = glob.glob1(dir,"*.bin");
    
    frame = 0;
    for file in files:
        frame = max(frame,int(file.split(".")[0].split("_")[1]));
        
    tag = '_' + '%03d' % frame + '.bin';
    
    fn_u = dir + '/' + 'u' + tag;
    fn_v = dir + "/" + 'v' + tag;
    fn_rho = dir + "/" + 'rho' + tag;
       
    u = np.fromfile(fn_u, dtype="double");
    v = np.fromfile(fn_v, dtype="double");
    rho = np.fromfile(fn_rho, dtype="double");
        
    if (cuda) : 
        u = np.reshape(u, (NY,NX), order='F');
        v = np.reshape(v, (NY,NX), order='F');
        rho = np.reshape(rho, (NY,NX), order='F');
    else:
        u = np.reshape(u, (NY,NX));
        v = np.reshape(v, (NY,NX));
        rho = np.reshape(rho, (NY,NX));
        
    u = u[1:-1,1:-1] / ulid;
    v = v[1:-1,1:-1] / ulid;
    rho = rho[1:-1,1:-1];
    
    dM = np.sum(rho.flatten()) - (NY - 2) * (NX - 2);
    print('dM = %.3e' % dM)
    
    ucenter = np.zeros(NY);
    ucenter[NY-1] = 1.0; 
    if (NX % 2 == 0):
        j = (NX - 2)//2 - 1;
        ucenter[1:-1] = 0.5*(u[:,j] + u[:,j + 1]);
    else:
        j = (NX - 1)//2 - 1;
        ucenter[1:-1] = u[:,j];
    vcenter = np.zeros(NX);
    if (NY % 2 == 0):
        i = (NY - 2)//2 - 1;
        vcenter[1:-1] = 0.5*(v[i,:] + v[i + 1,:]);
    else:
        i = (NY - 1)//2 - 1;
        vcenter[1:-1] = v[i, :];
        
    y = np.zeros(NY);
    y[NY-1] = 1.0;
    y[1:-1] = np.linspace(0.5, (NY-2) - 0.5, (NY-2)) / (NY - 2);
    
    x = np.zeros(NX);
    x[NX-1] = 1.0;
    x[1:-1] = np.linspace(0.5, (NX-2) - 0.5, (NX-2)) / (NX - 2);
    
    plt.figure(1)
    plt.plot(ucenter,y,'b-',linewidth=2.5);
    if (re == 100) : 
        plt.plot(data[:,2],data[:,0],'ko')
    elif (re == 400) : 
        plt.plot(data[:,4],data[:,0],'ko')
    elif (re == 1000) :
        plt.plot(data[:,6],data[:,0],'ko')
    elif (re == 3200) :
        plt.plot(data[:,8],data[:,0],'ko')
    elif (re == 5000) :
        plt.plot(data[:,10],data[:,0],'ko')
    elif (re == 5000) :
        plt.plot(data[:,12],data[:,0],'ko')
    elif (re == 7500) :
        plt.plot(data[:,14],data[:,0],'ko')
    elif (re == 10000) :
        plt.plot(data[:,16],data[:,0],'ko')
                
    plt.figure(2)
    plt.plot(x,vcenter,'b-',linewidth=2.5);
    if (re == 100) : 
        plt.plot(data[:,1],data[:,3],'ko')
    elif (re == 400) : 
        plt.plot(data[:,1],data[:,5],'ko')
    elif (re == 1000) :
        plt.plot(data[:,1],data[:,7],'ko')
    elif (re == 3200) :
        plt.plot(data[:,1],data[:,9],'ko')
    elif (re == 5000) :
        plt.plot(data[:,1],data[:,11],'ko')
    elif (re == 5000) :
        plt.plot(data[:,1],data[:,13],'ko')
    elif (re == 7500) :
        plt.plot(data[:,1],data[:,15],'ko')
    elif (re == 10000) :
        plt.plot(data[:,1],data[:,17],'ko')
                
plt.figure(1)
plt.legend(['lbm', 'Ghia et al. (1982)'],fontsize=18)

plt.figure(2)
plt.legend(['lbm', 'Ghia et al. (1982)'],fontsize=18)

fig1.savefig(str(NY) + "x" + str(NX) + "_u.png",bbox_inches='tight');
plt.close(fig1)

fig2.savefig(str(NY) + "x" + str(NX) + "_v.png",bbox_inches='tight');
plt.close(fig2)

