import numpy as np
import os
import random
from tqdm import tqdm
class IDKCluster:
    def __init__(self,cluster_id):
        # value is nc*\sum_{i \in Cluster}\phi(g|S)
        self.value = np.zeros(256, dtype=np.uint32)

        # nc num of trajectories in the cluster
        self.nc = 0

        self.trajectory = []

        # self owned cid (not real cid)
        self.cid = cluster_id
    def add_unit256(self,int_unit256):
        for i in range(256):
            self.value[i] = self.value[i] + int_unit256.get_bit(i)
    # usage: add_trajectory(trajectory_id, trajectory_to_rkhs2[trajectory_id])
    def add_trajectory(self, trajectory_id, trajectory_idk2):
        self.trajectory.append(trajectory_id)
        self.add_unit256(trajectory_idk2)
        self.nc = self.nc + 1
        
    def get_dot_product(self,int_unit256):
        r = 0.0
        for i in range(256):
            r = r + self.value[i] * int_unit256.get_bit(i)
        return r / self.nc
    
    def get_trajectory(self):
        return self.trajectory

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

# read data saved in IDK1.txt
def read_rkhs1_data():
    data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"IDK1.txt")

    rkhs1_data = []
    with open(data_path, 'r') as file:
        for line in file:
            if "idk:" in line:
                parts = line.split("idk:")[0].strip()
                idk_part = line.split("idk:")[1].strip()
                tid = int(parts.split(",")[1].split(":")[1])
                cid = int(parts.split(",")[2].split(":")[1])
                idk_values_list = [float(value) for value in idk_part.split(',') if value]
                idk_values_list.append(tid)
                idk_values_list.append(cid)
                rkhs1_data.append(idk_values_list)
    
    data = np.array(rkhs1_data, dtype=np.double)
    return data

# map trajectories (points in rkhs1) to rkhs2
def mapping_trajectory_to_rkhs2(data,t,phi):
    l = len(data)
    assert (l >= phi)

    trajectory_to_rkhs2 = np.empty(l, dtype=object)
    for i in range(l):
        trajectory_to_rkhs2[i] = Uint256()
    
    for i in tqdm(range(t), desc="sampling", unit="sample"):
        # sample phi points
        points_in_Voronoi_diagram = random.sample(range(l), phi)

        for j in tqdm(range(l), desc="map trajectories (points in rkhs1) to rkhs2", unit="point in rkhs1"):
            k = idxmin(data,points_in_Voronoi_diagram,j)
            trajectory_to_rkhs2[j].set_bit(i*phi+k)

    return trajectory_to_rkhs2
# d-dim argmin function
def idxmin(data,sample_points_index,target_point_index):
    l = len(sample_points_index)
    dis = -1.0
    k = -1
    _ , dim = data.shape

    for i in range(l):
        distance_metric = 0
        for s in range(dim-2):
            d = data[target_point_index,s]-data[sample_points_index[i],s]
            distance_metric = distance_metric + d*d

        if((distance_metric <= dis) or (dis < 0)):
            k = i
            dis = distance_metric
    return k

def get_num_of_clusters(data):
    # return k
    return len(np.unique(data[:,-1]))

def get_init_point_in_cluster(data,k,idx):
    # choose ids in data for clusters
    l = []
    for i in range(k):
        l.append(random.randint(idx[i], idx[i+1] - 1))
    return l

def get_argmax(N,clusters,trajectory_to_rkhs2):
    l = len(clusters)

    g = random.choice(N)

    dot_product = -1
    # find best cluster id
    k = -1
    for i in range(l):
        dp = clusters[i].get_dot_product(trajectory_to_rkhs2[g])
        if(dp > dot_product):
            dot_product = dp
            k = i

    return dot_product,k,g

def get_tau(N,clusters,trajectory_to_rkhs2):
    l = len(clusters)
    dot_product = -1

    for g in N:
        for i in range(l):
            dp = clusters[i].get_dot_product(trajectory_to_rkhs2[g])
            if(dp > dot_product):
                dot_product = dp

    return dot_product


if __name__ == "__main__":
    t = 16 # ~
    phi = 16 # ~ 
    # some os needs absdir...
    D = read_rkhs1_data()
    trajectory_to_rkhs2 = mapping_trajectory_to_rkhs2(D,t,phi)

    # N stores index
    N = list(range(len(D)))
    
    # estimate k
    num_of_cluster = get_num_of_clusters(D)
    

    # prehandle
    index = []
    prv = -1
    for i in range(len(D)):
        if(int(D[i,-1])!=prv):
            index.append(i)
            prv = prv + 1
    index.append(len(D))
    print(index)


    init_point_in_cluster = get_init_point_in_cluster(data=D,k=num_of_cluster,idx=index)

    # prepare to run the clustring algorithm
    C = np.empty(num_of_cluster, dtype=object)
    for i in range(num_of_cluster):
        C[i] = IDKCluster(i)
        trajectory_id = init_point_in_cluster[i]
        C[i].add_trajectory(trajectory_id, trajectory_to_rkhs2[trajectory_id])
        N.remove(trajectory_id)

    rho = 0.99
    tau = get_tau(N,C,trajectory_to_rkhs2)

    while(len(N) != 0 and tau >= 0.00001):
        tau = tau * rho

        dot_product,k,g = get_argmax(N,C,trajectory_to_rkhs2)
        C[k].add_trajectory(g, trajectory_to_rkhs2[g])
        N.remove(g)
    
    while(len(N) != 0):
        dot_product,k,g = get_argmax(N,C,trajectory_to_rkhs2)
        C[k].add_trajectory(g, trajectory_to_rkhs2[g])
        N.remove(g)

    # we are done....
    
    

    correct = 0
    wrong = 0
    for i in range(len(C)):
        l = C[i].get_trajectory()
        print("Cluster {}:{}".format(i,l))
        c = len([num for num in l if index[i] <= num <= index[i+1]])
        w = len(l) - c
        correct = correct + c
        wrong = wrong + w

    print("Accuracy:{}".format(correct/(correct+wrong)))

    


    
    
    