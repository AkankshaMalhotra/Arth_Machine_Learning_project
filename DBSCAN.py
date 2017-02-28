'''
@author: akanksha
'''
import matplotlib.pyplot as plt
import cluster
import numpy as np

class Dbscan():
    def __init__(self, data):
        self.cluster_count=0
        self.visited_points=[]
        self.clust_pts=[]
        self.dataset=data

    def dbscan(self, MinPts, eps):
        '''

        :param data: The input data points or features
        :param MinPts:
        :param eps:
        :return:
        '''
        self.cluster_count=0
        Noise=cluster.cluster("Noise")
        for point in self.dataset:
            if point not in self.visited_points:
                self.visited_points.append(point)
            else:
                continue
            neighbour_points=self.region_query(point, eps)
            if len(neighbour_points) < MinPts:
                Noise.add(point)
            else:
                cluster_name="cluster "+ str(self.cluster_count)
                C=cluster.cluster(cluster_name)
                self.cluster_count+=1
                self.expand_cluster(point, neighbour_points, C, eps, MinPts)
                plt.plot(C.get_syllable(), C.get_usage(), 'o', label=cluster_name, hold = True)

        if len(Noise.getPoints()) != 0:
            plt.plot(Noise.get_syllable(), Noise.get_usage(), 'x', label='Noise', hold = False)
        plt.show()

    def expand_cluster(self, point, neighbour_points, C, eps, MinPts):
        C.add(point)
        for p in neighbour_points:
            if p in self.visited_points:
                continue
            else:
                self.visited_points.append(p)
            neighbour= self.region_query(p, eps)
            if len(neighbour) >= MinPts:
               for n in neighbour:
                   if n in neighbour_points:
                       continue
                   neighbour_points.append(n)
            self.clust_pts+= C.getPoints()
            if p not in self.clust_pts:
                C.add(p)
                self.clust_pts.append(p)

    def region_query(self, points, eps):
        dist=np.linalg.norm(np.asarray(self.dataset)-points, axis=1)
        result=[self.dataset[i] for i, j in enumerate(dist) if j <= eps]
        return(result)



