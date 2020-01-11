import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "-re",
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
        "-curl",
        type=int,
        default=1,
    )

parser.add_argument(
        "-dir",
        type=str,
        default="output",
    ) 
args = parser.parse_args();
 
re = args.re;
NX = args.nx;
NY = args.ny;
ulid = args.ulid;
cuda = args.cuda;
plot_curl = args.curl;
dir = args.dir;

files = glob.glob1(dir,"*.bin");

max_frame = 0;
for file in files:
    max_frame = max(max_frame,int(file.split(".")[0].split("_")[1]));

fig1 = plt.figure(1,figsize=(4,4), dpi=100)
plt.axis('off')
img = plt.imshow(np.zeros((NY,NX)),cmap='coolwarm',interpolation='nearest');
plt.axis('off')
img.axes.get_xaxis().set_visible(False)
img.axes.get_yaxis().set_visible(False)

for frame in range(0,max_frame + 1):
       
    tag = '_' + '%03d' % frame + '.bin';
    
    fn_u = dir + '/' + 'u' + tag;
    fn_v = dir + "/" + 'v' + tag;
       
    u = np.fromfile(fn_u, dtype="double");
    v = np.fromfile(fn_v, dtype="double");
        
    if (cuda) : 
        u = np.reshape(u, (NY,NX), order='F');
        v = np.reshape(v, (NY,NX), order='F');
    else:
        u = np.reshape(u, (NY,NX));
        v = np.reshape(v, (NY,NX));
        
    u = u[1:-1,1:-1] / ulid;
    v = v[1:-1,1:-1] / ulid;
        
    if (plot_curl):
        grad_u = np.gradient(u,edge_order=2);
        grad_v = np.gradient(v,edge_order=2);
        u_y = grad_u[1];
        v_x = grad_v[0];
        curl = v_x - u_y;

    else:
        mag = np.sqrt(u**2 + v**2);
    
    plt.figure(1);
    if (plot_curl):
        img.set_data(np.flipud(curl));
        img.set_clim(vmin=-0.005,vmax=0.005);
        plt.savefig("re" + str(int(re)) + "_" + str(NY) + "x" + str(NX) + "_curl_" + "%03d" % frame + ".png",bbox_inches='tight',pad_inches=0);

    else:
        img.set_data(np.flipud(mag));
        img.set_clim(vmin=0, vmax = 1)
        plt.savefig("re" + str(int(re)) + "_" + str(NY) + "x" + str(NX) + "_mag_" + "%03d" % frame + ".png",bbox_inches='tight',pad_inches=0);
