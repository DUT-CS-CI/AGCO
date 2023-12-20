#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
# 

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans

default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # parameters of the "mutation" operator
        "sigma_iso": 0.01,
        # parameters of the "cross-over" operator
        "sigma_line": 0.2,
        # when to write results (one generation = one batch)
        "dump_period": 2500,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": np.zeros(15),
        "max": np.ones(15),
        # iso variation
        "iso_sigma": 1./300.,
        "line_sigma": 20./300.
    }
class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid

def scale(x,params):
    assert(params["max"].shape[0] >= x.shape[0])
    assert(params["min"].shape[0] >= x.shape[0])
    return x * (params["max"] - params["min"]) + params["min"] 
    
def random_individual(dim_x, params):
    x = np.random.random(dim_x)
    x = scale(x, params)
    return x.clip(params["min"], params["max"])

# 传入cross_factor

def variation_xy(x, z, params, cross_factor):
    assert(x.shape == z.shape)
    assert(params["max"].shape[0] >= x.shape[0])
    assert(params["min"].shape[0] >= x.shape[0])
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params["iso_sigma"] * (p_max - p_min), size=len(x))
    b = np.random.normal(0, params["line_sigma"] * (p_max - p_min), size=len(x))
    #原来的算子
    y = np.zeros(len(x))
    if(cross_factor == 0):
        y = x.copy() + a + b * (x - z) # 产生后代
    #均匀交叉算子
    if(cross_factor == 1):
        dim = x.shape[0]
        for i in range(0, dim):
            rand = random.random()
            if(rand > 0.5):
                y[i] = x[i]
            else:
                y[i] = z[i]
        
    #算术交叉算子
    if(cross_factor == 2):
        dim = x.shape[0]
        rand = random.random()
        eta = 1/(1 + 0.7) #那个eta设为0.7
        if(rand > 0.5):
            beta = (2 * rand) ** eta
        else:
            beta =  (1/ (2 - rand * 2)) ** eta
        for i in range(0, dim):
            y[i] = 0.5 *((1 + beta) * x[i] + (1 - beta) * z[i])
        
    y = np.clip(y, p_min, p_max)
    return y 


def variation(x, archive, params,cross_factor):
  keys = list(archive.keys())
  z = archive[keys[np.random.randint(len(keys))]].x #另一个个体
  return variation_xy(x, z, params,cross_factor)

# 生成文件名：centroids_5000_2.dat
def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'

def __datas_filename(k, dim):
    return 'datas_' + str(k) + '_' + str(dim) + '.dat'

def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')

def __write_datas(datas):
    k = datas.shape[0]
    dim = datas.shape[1]
    filename = __datas_filename(k, dim)
    with open(filename, 'w') as f:
        for p in datas:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True,dim_x=None):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:# 如果先前生成了该文件，直接load 即可
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim) # samples行dim列
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means.fit(x) # 计算k 个簇
    __write_centroids(k_means.cluster_centers_) #k 个簇记录下来

    x2=k_means.cluster_centers_
    
    cluster_number=int(k/5)
    k_means2 = KMeans(init='k-means++', n_clusters=cluster_number,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means2.fit(x2)

    classx2 = k_means2.predict(x2)#返回聚类结果
    c2=classx2.reshape(k,1)
    data=np.hstack((x2,c2))#将聚类结果与第一次聚类中心拼接
    
    # 后面加上dim_x + 1 的空间！！！！！！！！！！！
    append=np.zeros((k,1+dim_x)) #当输入为3时会出现 datas_5000_7.dat结果

    #随机选择一种交叉因子，0,1,2
    cross_factor  =   np.zeros((k,1))
    for i in range(k):
        cross_factor[i,0] = random.randint(0,2)    

    data=np.hstack((data,append))
    data = np.hstack((data,cross_factor))
    data = data[np.argsort(data[:,2])]#按聚类结果排序，并返回ndarray新数组
    __write_datas(data)#新的到的结果写入文件，也可以不写

    return k_means.cluster_centers_, data, cluster_number

def cvt2(k, dim, centroid, cvt_use_cache=True, dim_x=None):
    fname = __datas_filename(k, dim)
    if cvt_use_cache:# 如果先前生成了该文件，直接load 即可
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    cluster_number=int(k/5)
    k_means2 = KMeans(init='k-means++', n_clusters=cluster_number,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means2.fit(centroid) # 计算k 个簇
    classx2 = k_means2.predict(centroid)#返回聚类结果，就是centroid 分到第几类了 
    c2=classx2.reshape(k,1)

    # 引入taskIndex ，为各个任务标号
    a = np.arange(k).reshape(k,1)
    taskIndex = np.hstack((a,c2))
    #print(type(centroid[0][0]))
    # 将聚类结果与第一次聚类中心拼接
    data=np.hstack((centroid,c2))
    #print(type(data[0][0]))
    #print(type(c2[0][0]))
    # 后面加dim_x个空间
    append = np.zeros((k,38))
    data=np.hstack((data,append))

    # 按聚类结果排序，并返回ndarray新数组
    data = data[np.argsort(data[:,dim])]

    print(data)
    # taskIndex也是按第二列排序，它的作用就是确定第几个任务是第几类
    taskIndex = taskIndex[np.argsort(taskIndex[:,1])]

    print('cvt2run')
    __write_datas(data)#新的到的结果写入文件，也可以不写
    __write_datas(taskIndex)

    return data, taskIndex,cluster_number



def make_hashable(array):
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):
    def write_array(a, f): # 把a 的每一项都写进f 中
        for i in a:
            f.write(str(i) + ' ')
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ') # 只有fitness 不是向量，其他的都是dim 维的向量
            write_array(k.centroid, f)
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")
