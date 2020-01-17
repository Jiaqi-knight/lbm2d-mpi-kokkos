import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
        "-nx",
        type=int,
    )
 
parser.add_argument(
        "-ny",
        type=int,
    )

parser.add_argument(
        "-dir",
        type=str,
        default="output",
    ) 

parser.add_argument(
        "-cuda",
        type=int,
        default=0,
    )

parser.add_argument(
        "-rm",
        type=int,
        default=0,
    )
args = parser.parse_args();
 
NX = args.nx;
dir = args.dir;
cuda = args.cuda;
remove_old = args.rm;

files = glob.glob1(dir,"*.bin");

max_frame = 0;
max_rank = 0;

for file in files:
    strings = file.split(".")[0].split("_");

    max_frame = max(max_frame,int(strings[1]));
    max_rank = max(max_rank, int(strings[3]));

rem = (args.ny - 2) % (max_rank + 1);

for frame in range(0,max_frame + 1):

    tag_comb = '_' + '%03d' % frame + '.bin';
    
    u_out = open(dir + "/" + "u" + tag_comb, "ab")
    v_out = open(dir + "/" + "v" + tag_comb, "ab")
    rho_out = open(dir + "/" + "rho" + tag_comb, "ab")
    
    for rank in range(0,max_rank + 1):
        
        tag = '_' + '%03d' % frame + "_rank_" + "%03d" % rank + '.bin';
        
        if (rank < rem):
            NY = (args.ny - 2)//(max_rank + 1) + 1 + 2;
        else:
            NY = (args.ny - 2)//(max_rank + 1) + 2;
                   
        u_in = dir + '/' + 'u' + tag;
        v_in = dir + "/" + 'v' + tag;
        rho_in = dir + "/" + 'rho' + tag;
        
        u = np.fromfile(u_in, dtype="double");
        v = np.fromfile(v_in, dtype="double");
        rho = np.fromfile(rho_in, dtype="double");

        # convert cuda to row major
        if (cuda) : 
            u = np.reshape(u, (NY,NX), order='F');
            v = np.reshape(v, (NY,NX), order='F');
            rho = np.reshape(rho, (NY,NX), order='F');
            u = u.flatten();
            v = v.flatten();
            rho = rho.flatten();
            
        # remove top ghost layer
        if (rank == 0):
        
            u = u[0:-NX];
            v = v[0:-NX];
            rho = rho[0:-NX];
        
        elif (rank == max_rank):
            
            u = u[NX::];
            v = v[NX::];
            rho = rho[NX::];
            
        else: 
            
            u = u[NX:-NX];
            v = v[NX:-NX];
            rho = rho[NX:-NX];   
                         
        u.tofile(u_out,sep="");
        v.tofile(v_out,sep="");
        rho.tofile(rho_out,sep="");

        if (remove_old):
            
            os.remove(u_in);
            os.remove(v_in);
            os.remove(rho_in);