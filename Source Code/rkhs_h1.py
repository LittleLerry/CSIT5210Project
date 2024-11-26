import numpy as np
import os
import random
from tqdm import tqdm
import sys

class Uint256:
    def __init__(self):
        self.value = np.zeros(4, dtype=np.uint64)

    def set_bit(self, k):
        if not (0 <= k < 256):
            raise ValueError("It should not happen. k out of range.")
        index = k // 64
        bit_position = k % 64 
        self.value[index] |= np.uint64(1 << bit_position)
    def to_binary(self):
        return ''.join(format(x, '064b') for x in self.value[::-1])
    def get_bit(self,k):
        if not (0 <= k < 256):
            raise ValueError("It should not happen. k out of range.")
        index = k // 64
        bit_position = k % 64 
        mask = np.uint64(1 << bit_position)
        return 1 if (self.value[index] & mask) != 0 else 0

class RKHSTrajectory:
    def __init__(self):
        self.value = np.zeros(256, dtype=np.uint32)
        self.nc = 0
        self.trajectory = -1
        self.cluster = -1
    def add_unit256(self,int_unit256):
        self.nc = self.nc + 1
        for i in range(256):
            self.value[i] = self.value[i] + int_unit256.get_bit(i)
    def set_trajectory(self,id):
        self.trajectory = id
    def set_cluster(self,id):
        self.cluster = id
    def get_trajectory(self):
        return self.trajectory
    def get_cluster(self):
        return self.cluster
    def to_string(self):
        result = ""
        if self.nc != 0 and self.cluster != -1:
            result += f"nc:{self.nc},"
            result += f"tid:{self.trajectory},"
            result += f"cid:{self.cluster},"
            result += "idk:"
            for i in range(256):
                result += f"{self.value[i] / self.nc},"
            result += "\n"
        return result



def read_data(data_dir,cluster_restriction = 12):
    temp_list = []
    trajectory = int(0)
    for i in tqdm(range(cluster_restriction), desc="processing", unit="dir"):
        dir_name = f"{i:03d}/Trajectory"
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.plt'):
                    file_path = os.path.join(dir_path, filename)
                    with open(file_path, 'r') as file:
                        for _ in range(6):
                            next(file)

                        for line in file:
                            fields = line.strip().split(',')
                            # read all data
                            if len(fields) >= 2:
                                latitude = float(fields[0])
                                longitude = float(fields[1])
                                cluster = int(i)
                                temp_list.append([latitude, longitude, cluster, trajectory])
                        trajectory = trajectory + 1

    data = np.array(temp_list)
    return trajectory , data


# We utilize a Voronoi diagram to seperate a given dataset. At each iteration, we perform $\phi$-random sampling 
# and use the extracted $\phi$ points to construct a Voronoi diagram in the given space. It is evident that 
# in non-degenerate situations, the space will be divided into $\phi$ regions, denoted as $S_1, S_2, \ldots, S_\phi$. 
# For a specific data point $x$, in the non-degenerate case, it will belong exclusively to one of the subspaces, 
# denoted as $S_{k}$. We define the vector $V_x = [e_1, \ldots, e_\phi]$, where $e_i = 1$ if $i = k$, and $e_i = 0$ 
# otherwise. In this manner, $V_x$ maps the point $x$ to Reproducing Kernel Hilbert Space. i.e. $V_x = \phi (x|S)$. 
# This operation can be repeated $t$ times. For a particular data point $x$, we concatenate the results from these 
# $t$ iterations to obtain an estimate of $x$ in RKHS. It is not essential to know the order of $\phi$ points in 
# each sample, trivially.

# Although calculating a Voronoi diagram in a $d$-dimensional space consumes $O(n \log n)$ time, we also need to 
# determine which region a specific point belongs to, which may require complex data structures. However, if 
# we consider the definition of a Voronoi diagram, a point $x$ belongs to the region $S_{j}$ if and only if it 
# is closest to the $j$th sample point. Therefore, in practice, it only takes $O(\phi)$ time to determine the 
# region to which the point $x$ belongs.
def mapping_point_to_rkhs1(data,t,phi):
    # we only have the conf for t*phi <= 256...
    l = len(data)
    assert (l >= phi)

    # init zero unit256
    point_to_rkhs1 = np.empty(l, dtype=object)
    for i in range(l):
        point_to_rkhs1[i] = Uint256()

    for i in tqdm(range(t), desc="sampling", unit="sample"):
        # sample phi points
        points_in_Voronoi_diagram = random.sample(range(l), phi)

        for j in tqdm(range(l), desc="mapping point to rkhs1", unit="point"):
            k = idxmin(data,points_in_Voronoi_diagram,j)
            point_to_rkhs1[j].set_bit(i*phi+k)

    return point_to_rkhs1
    

# Subsequently, we map each point within the trajectory and compute the average, which serves as the 
# estimation of the IDK embedding.
def mapping_trajectory_to_rkhs1(data,point_in_rkhs1,T):
    l = len(data)
    RKHSTrajectories = np.empty(T, dtype=object)
    for i in range(T):
        RKHSTrajectories[i] = RKHSTrajectory()

    for i in tqdm(range(l), desc="mapping trajectory to rkhs1", unit="points in trajectory"):
        trajectory_id = int(data[i][3])
        cluster_id = int(data[i][2])
        RKHSTrajectories[trajectory_id].add_unit256(point_in_rkhs1[i])
        RKHSTrajectories[trajectory_id].set_cluster(cluster_id)
        RKHSTrajectories[trajectory_id].set_trajectory(trajectory_id)
    
    return RKHSTrajectories

# argmin function
def idxmin(data,sample_points_index,target_point_index):
    l = len(sample_points_index)
    dis = -1.0
    k = -1
    for i in range(l):
        delta_x = data[target_point_index,0]-data[sample_points_index[i],0]
        delta_y = data[target_point_index,1]-data[sample_points_index[i],1]
        if((delta_x*delta_x + delta_y*delta_y <= dis) or (dis < 0)):
            k = i
            dis = delta_x*delta_x + delta_y*delta_y
    return k

def write_data(rkhs_trajectory):
    file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"IDK1.txt")

    l = len(rkhs_trajectory)

    with open(file_path, 'w') as file:
        for i in tqdm(range(l), desc="writing", unit="write trajectory"):
            if (rkhs_trajectory[i].get_cluster() != -1):
                file.write(rkhs_trajectory[i].to_string())

if __name__ == "__main__":
    t = 16 # ~
    phi = 16 # ~ 
    # some os needs absdir...
    data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Data")
    
    T, D = read_data(data_dir,int(input("Enter an integer (2 to 4): ")))

    point_in_rkhs1 = mapping_point_to_rkhs1(D,t,phi)
    rkhs_trajectories = mapping_trajectory_to_rkhs1(D,point_in_rkhs1,T)
    write_data(rkhs_trajectories)