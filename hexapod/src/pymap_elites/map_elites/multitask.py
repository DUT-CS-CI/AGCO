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
# from scipy.spatial import cKDTree : TODO
from scipy import stats
import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.neighbors import KDTree
from scipy.spatial import distance

from map_elites import common as cm


# TODO : we do not need the KD-tree here -> use the archive directly
# TODO : remove the to_evaluate_centroid


#优胜劣汰，把表现好的Species 加入到archieve。centroid 相当于各个任务
def add_to_archive(s, archive):
    # make_hashable：将centroid 变为float ?
    centroid = cm.make_hashable(s.centroid)
    if centroid in archive:
        if s.fitness > archive[centroid].fitness:
            archive[centroid] = s
            return 1
        return 0
    else:
        archive[centroid] = s
        return 1


#优胜劣汰，把表现好的Species 加入到archieve。centroid 相当于各个任务
def add_to_archive_data(s, archive,data,dim_x):
    # make_hashable：将centroid 变为float ?
    centroid = cm.make_hashable(s.centroid)
    tuple_class=np.where(data[:,0]==centroid[0])#可改进
    task_same=tuple_class[0][0]
    if centroid in archive:
        if s.fitness > archive[centroid].fitness:
            for j in range(0,dim_x):
                # 4变14
                data[task_same,j+14]=archive[centroid].x[j]
            archive[centroid] = s
            return 1
        return 0
    else:
        archive[centroid] = s
        for j in range(0,dim_x):
            #print(s.x[j])
            #print(task_same)
            #print(data.shape)
            data[task_same,j+14]=s.x[j]        
        return 1
# evaluate a single vector (z) with a function f and return a species
def add_to_archive_data_tompson(s, archive,data,dim_x):
    # make_hashable：将centroid 变为float ?
    centroid = cm.make_hashable(s.centroid)
    tuple_class=np.where(data[:,0]==centroid[0])#可改进
    task_same=tuple_class[0][0]

    if s.fitness > archive[centroid].fitness:
            a=archive[centroid].fitness
            for j in range(0,dim_x):
                # 4变14
                data[task_same,j+14]=archive[centroid].x[j]
            archive[centroid] = s
            action_type=data[task_same,14+dim_x]
            return 1,action_type,s.fitness-a
    return 0,-1,-1

# t = vector, function
def __evaluate(t):
    z, f, task, centroid, _ = t 
    fit = f(z, task)
    return cm.Species(z, task, fit, centroid)

# bandit opt for optimizing tournament size
# probability matching / Adaptive pursuit Thierens GECCO 2005
# UCB: schoenauer / Sebag
# TODO : params for values, and params for window
def bandit(successes, n_niches):
    n = 0
    for v in successes.values():
        n += len(v)
    v = [1, 10, 50, 100, 500]#, 1000]
    if len(successes.keys()) < len(v):
        return random.choice(v)
    ucb = []
    for k in v:
        x = [i[0] for i in successes[k]]
        mean = sum(x) / float(len(x)) # 100 = batch size??
        n_a = len(x)
        ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]
    a = np.argmax(ucb)
    t_size = v[a]
    return t_size

# select the niche according to
# 返回在t_size 个任务中和x(Species) 最相似的
def select_niche(n,f, centroids, tasks, t_size, params,datas,archive,keys,rand,use_distance=False,):
    x = archive[keys[rand[n]]]
    to_evaluate = []
    if not use_distance:

        cross_factor = random.randint(0,2)
        z = cm.variation(x.x, archive, params,cross_factor) # 产生的后代
        # No distance: evaluate on a random niche
        niche = np.random.randint(len(tasks))
        to_evaluate += [(z, f, tasks[niche], centroids[niche, :], params)]

    else:
        # we select the parent (a single one), then we select the niche
        # with a tournament based on the task distance
        # the size of the tournament depends on the bandit algorithm
        niches_centroids = []
        niches_tasks = [] # TODO : use a kd-tree
        # rand 是随机产生t_size 个数,范围是0 ~ 5000 
        rand = np.random.randint(centroids.shape[0], size=t_size) # 参数1 是最大值，参数2 是产生几个数字
        # for 是把这些数的centroid 和tasks 连接起来
        for p in range(0, t_size):
            n = rand[p]
            niches_centroids += [centroids[n, :]]
            niches_tasks += [tasks[n]]
        # 计算各个x.centroid 和niches_centroids 的距离,找最小的
        cd = distance.cdist(niches_centroids, [x.centroid], 'euclidean')
        cd_min = np.argmin(cd) # 挑出index of min distance
        # 找出了这个任务，把它加到to_evaluate 中
        # parent selection，选定一个parent
        # print(niches_tasks[cd_min])
        # print(datas[:,0])
        # copy & add variation
        tuple_class=np.where(datas[:,0]==niches_tasks[cd_min][0][0])#可改进
        # print(niches_tasks[cd_min])
        # print(tuple_class)
        task_same=tuple_class[0][0]
        cross_factor=datas[task_same][-1]
        z = cm.variation(x.x, archive, params,cross_factor) # 产生的后代   
        to_evaluate += [(z, f, niches_tasks[cd_min], niches_centroids[cd_min], params)]
    return to_evaluate
        
        



def add_to_count(archive,datas,pool,len_datas,cluster_number,f,params,dim_x,n_evals,num_evals,action,taskIndex,tasks,centroids,crossover_sum,thompson_params):
    to_evaluate = []
    task_class=int (cluster_number)
    nev=n_evals/num_evals
    action_num=np.zeros(10)#记录此次统一评估每种action选取了多少次
    cross_factor = 0
    change_sum = 0 # 遍历1000类后记录factor改变的总和
    for h in range(0,task_class):
        crossover_sum = np.array([0,0,0])
        #2 变 12
        tuple_class=np.where(datas[:,12]==h)# h 类任务的合集
        task_same=tuple_class[0]#task_same里面存放的是所有与最相似点聚类结果相同的任务点，形式是datas数组的下标,array 变成 list
        #print(tuple_class)
        centroid_sameclass=[]#该类所有点在archive上的坐标
        centroid_samex=[]
        data_index=[]#记录同组任务对应的data的位置
        count=0
        around_fitness = np.array([])
        old_fitness=np.array([])
        for i in task_same:
            # 2 变 12
            task_centroids=datas[i,0:12]
            # print(type(task_centroids[0]))
            centroid = cm.make_hashable(task_centroids)
            # print(type(centroid))
            # print(centroid)
            if centroid in archive:#只考虑已经存放在archive内的任务
                count+=1
                # 3 变 13
                old_fitness=np.append(old_fitness,datas[i,13])#该类任务旧的适应度

                new_fitness=archive[centroid].fitness
                datas[i,13]=new_fitness
                data_index.append(i)                     
                # 12维叠加起来
                for l in range(12):
                    centroid_sameclass.append((np.float(task_centroids[l])))#该类任务的质心坐标集合
                for j in range(0,dim_x):
                # 4 变 14
                    centroid_samex.append(np.float(datas[i,14+j]))
                    
                #print(centroid_samex)
                # print(type(centroid_sameclass[0]))
                around_fitness = np.append(around_fitness, new_fitness) #该类任务新的适应度   
            #print(centroid_samex)         
        if count!=0 :   
            index = np.arange(0,count)
            not_same=index[old_fitness!=around_fitness]
            same=index[old_fitness==around_fitness] #目的是找到该类任务中，在这两千次评估内得以改进的任务。返回的是old_fitness或around_fitness内的下标



            biggest=-9999;
            arg_biggest=-1;#####################此轮没有改进过的任务若满足要求下一轮必定被改进，是否有必要？
            for a in not_same:
                if around_fitness[a]>biggest:
                    biggest=around_fitness[a]
                    arg_biggest=a

                if datas[data_index[a]][-1] == 0:
                    crossover_sum[0] = crossover_sum[0] + 1
                elif datas[data_index[a]][-1] == 1:
                    crossover_sum[1] = crossover_sum[1] + 1
                elif datas[data_index[a]][-1] == 2:
                    crossover_sum[2] = crossover_sum[2] + 1


            need_change=[] 
            index_fitness_low=np.argsort(around_fitness)
            index_fitness_high=np.argsort(-around_fitness)
            arg_lowest=index_fitness_low[0]
            arg_biggest_all=index_fitness_high[0]
            biggest_all=around_fitness[arg_biggest_all]
            lowest=around_fitness[arg_lowest]
            #disparity=0.8*(biggest_all-lowest)
            # around_num=len(around_fitness)
            centroid_biggest_all=[]
            for j in range(0,12):
                centroid_biggest_all.append(centroid_sameclass[12*arg_biggest_all+j])
            c_biggest_all= cm.make_hashable(centroid_biggest_all)



            for a in same:#在改进的任务中寻找适应度最高的任务，与未改进的任务做比较
                if biggest-around_fitness[a]>0.01:
                    need_change=np.append(need_change,a)
            #if(len(index_fitness_low)<2):
            #need_change=np.append(need_change,index_fitness_low[0])
            #else:
                #for d in range(0,2):
                    #need_change=np.append(need_change,index_fitness_low[d]) 
            #for a in index_fitness_low:
            #for d in range(0,around_num):
                # if biggest-around_fitness[a]>disparity:
                # need_change=np.append(need_change,a) 
            tag=1
            if(len(need_change)!= 0):

                for b in need_change:
                    b=np.int(b)
                    arg_biggest=np.int(arg_biggest)

                    centroid_i=[]
                    x_old=[]
                    for j in range(0,12):
                        centroid_i.append(centroid_sameclass[12*b+j])
                    b_index=data_index[b]                  
                    c_i= cm.make_hashable(centroid_i)
                    act=[]
                    #开始进行采样
                    for ac in range (0,10):
                        act.append(stats.beta.rvs(action[ac,0],action[ac,1]))#即基于每种动作的贝塔分布进行采样，使该任务采用采样值最大的更新策略
                    
                    max_index = act.index(max(act, key = abs)) 

                    if max_index<=5 :          
                        for j in range(0,dim_x):
                            x_old.append(centroid_samex[dim_x*b+j])  
                        x1=archive[c_i].x+thompson_params[max_index][0]*(archive[c_i].x-x_old)
                        x=x1
                        datas[b_index][14+dim_x]=max_index
                        action_num[max_index]=action_num[max_index]+1
                        for j in not_same:
                            if j==arg_biggest:
                                tag=0
                            centroid_j=[]
                            for mm in range(0,12):
                                centroid_j.append(centroid_sameclass[12*j+mm])
                            c_j= cm.make_hashable(centroid_j)                     
                            x=x+thompson_params[max_index][1]*(archive[c_j].x-x1)
                        if tag==1:
                            x=x+thompson_params[max_index][1]*(archive[c_biggest_all].x-x1)


                    else:                                      
                        x1=archive[c_i].x
                        datas[b_index][14+dim_x]=max_index
                        action_num[max_index]=action_num[max_index]+1                  
                        for j in range(0,dim_x):   
                            x1[j]=x1[j]+np.random.normal(0, thompson_params[max_index][0])
                        x=x1

                   
                    # print(data_index[b])
                    # print(taskIndex[data_index[b]][0])
                    # print(tasks[taskIndex[data_index[b]][0]])
                    # print(centroids[taskIndex[data_index[b]][0]])
                    for xn in range(0,36):
                        if x[xn]>1:
                            x[xn]=1
                        if x[xn]<0:
                            x[xn]=0
                        
                    #print(x)
                    to_evaluate += [(x, f, tasks[taskIndex[data_index[b]][0]], centroids[taskIndex[data_index[b]][0]], params)]
            
            # 找出cross_sum最大值的索引
            rand = random.random()
            # 设定一个随机因子0.7，不能老选最多那个
            if(rand > 0.2):
                # 到目前为止，选择次数最大的那个factor
                      cross_factor = np.where(crossover_sum==np.max(crossover_sum))
                      cross_factor = cross_factor[0][0]
            else:
                cross_factor = random.randint(0,2)

            # 更新datas里待改进任务的factor值
            for b in need_change:
                b=np.int(b)
                datas[data_index[b]][-1] = cross_factor
            change_sum = change_sum + len(need_change)
    # print("形成的cross——sum为")
    # print(crossover_sum)
    print("改变的总和为: ",change_sum)

    # 最后返回的factor 是遍历过1000组的总的选择的次数
    return to_evaluate,action_num,cross_factor



def compute(dim_map=-1, 
            dim_x=-1, 
            f=None, 
            num_evals=1e5, 
            centroids=[],
            tasks=[], 
            variation_operator=cm.variation,
            params=cm.default_params,
            log_file=None,
            datas=None,
            cluster_number=None,
            taskIndex = None
            ):
    """Multi-task MAP-Elites
    - if there is no centroid : random assignation of niches
    - if there is no task: use the centroids as tasks
    - if there is a centroid list: use the centroids to compute distances
    when using the distance, use the bandit to select the tournament size (cf paper):

    Mouret and Maguire (2020). Quality Diversity for Multitask Optimization
    Proceedings of ACM GECCO.
    """
    print(centroids)
    len_datas=len(datas)
    judge=0
    print(params)
    assert(f != None)
    assert(dim_x != -1)
    # handle the arguments
    use_distance = False
    if tasks != [] and centroids != []:
        use_distance = True
    elif tasks == [] and centroids != []:
        # if no task, we use the centroids as tasks
        tasks = centroids
        use_distance = True
    elif tasks != [] and centroids == []:
        # if no centroid, we create indices so that we can index the archive by 
        #两个都没有时新建centroid
        centroids = np.arange(0, len(tasks)).reshape(len(tasks), 1)
        use_distance = False
    else:
        raise ValueError('Multi-task MAP-Elites: you need to specify a list of task, a list of centroids, or both')
    print("Multitask-MAP-Elites:: using distance =>", use_distance)

    assert(len(tasks) == len(centroids))
    n_tasks = len(tasks)

    # init archive (empty)
    archive = {}

    init_count = 0 # 初始化的个体数

    # init multiprocessing
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # main loop
    n_evals = 0 # number of evaluations
    b_evals = 0 # number evaluation since the last dump
    t_size = 1  # size of the tournament (if using distance) [will be selected by the bandit]
    successes = defaultdict(list) # count the successes
 #下面构建汤普森采样的初始beta分布,存储不同任务的初始参数
    action=np.zeros((10,2))#存放控制贝塔分布的参数，4*2 4代表4种粒子运动方式。1：粒子自身历史方向影响较大 2：粒子受同组其他任务影响较大 3：粒子每维增加较大随机值 4：粒子每维增加较小随机值
    thompson_params=np.zeros((10,2))
    param1=0.5
    param2=1.5
    param3=0.02

    nev=0#无用
    nev_old=nev#无用
    #print(action)
    #设置初始参数，即最开始使1与2发生的可能性最大，3，4发生可能性很小
    for i in range(6):
        action[i][0]=1.75 #第一位代表方式，第二位代表β和α，0是α，1是β
        action[i][1]=0.5
        thompson_params[i][0]=param1
        thompson_params[i][1]=param2
        param2=param2-0.2
        param1=param1+0.2

    for i in range(6,10):
        action[i][0]=0.5
        action[i][1]=1.75
        thompson_params[i][0]=param3
        param3=param3+0.02

    # 初始化cross_factor 为0
    cross_factor = 0
    while (n_evals < num_evals):
        to_evaluate = []
        to_evaluate1=[]
        to_evaluate_centroid = []
        # "random_init": 0.1，开始前要填充的tasks 份额
        if n_evals == 0 or init_count<=params['random_init'] * n_tasks:

            # initialize the map with random individuals
            # random_init_batch：100 ：一次初始化的数目
            for i in range(0, params['random_init_batch']):
                # create a random individual，生成一个dim 维的x
                x = cm.random_individual(dim_x, params)
                # we take a random task，从n 个task 中选一个
                n = np.random.randint(0, n_tasks)
                #list 之间的相加，即100 个参数叠在一起的list
                to_evaluate += [(np.array(x), f, tasks[n], centroids[n], params)]

            # to_evaluate 给 __evaluate 提供参数，
            # __evaluate 是用f 评价x，并返回一个Species，s_list 由Species 组成
            
            
            s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
            
            
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)

            # 各个好的Species 放入archive[centroid] 对应的位置中
            for i in range(0, len(list(s_list))):
                add_to_archive_data(s_list[i], archive,datas, dim_x)
            init_count = len(archive)
        else:

            #!!!!!!!!!!!!引入cross数组，每过一轮就更新一下
            crossover_sum = np.array([0,0,0])  
            # main variation/selection loop，keys 是已有的坐标
            keys = list(archive.keys())
            # we do all the randint together because randint is slow
            # rand 是随机生成的100 个[0,len(keys)] 的序号
            rand = np.random.randint(len(keys), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                  to_evaluate += select_niche(n,f, centroids, tasks, t_size, params,datas,archive,keys,rand,use_distance)

            # parallel evaluation of the fitness
            s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            # natural selection
            suc = 0
            for i in range(0, len(list(s_list))):
                suc += add_to_archive_data(s_list[i], archive,datas, dim_x)
            if use_distance:
                successes[t_size] += [(suc, n_evals)]
        if use_distance: # call the bandit to optimize t_size
            t_size = bandit(successes, n_tasks)
            



        # write archive
        if params['dump_period'] != -1 and b_evals > params['dump_period']:
            crossover_sum = np.array([0,0,0])
            cm.__save_archive(archive, n_evals)
            b_evals = 0 # 下一批
            action_suc=np.zeros(10)           
            #action_successes_num=np.zeros(10)   
           
            # !!!!!!!!!!!!!!!!!!!!传出cross_factor                           
            to_evaluate1,action_num,cross_factor=add_to_count(archive,datas,pool,len_datas,cluster_number,f,params,dim_x,n_evals,num_evals,action,taskIndex,tasks,centroids,crossover_sum,thompson_params)
            #print(cross_factor)            
            
            
            n_evals += len(to_evaluate1)
            b_evals += len(to_evaluate1)
            #print(len(to_evaluate1))
            s_list1 = cm.parallel_eval(__evaluate, to_evaluate1, pool, params)
            total_success=0
            for i in range(0, len(list(s_list1))):
                is_suc,action_type,ac_fitness=add_to_archive_data_tompson(s_list1[i], archive,datas, dim_x)#新函数,主要目的是获取该任务在统一评估时选取的action方法，与通过该方法后fitness进步了多少
                print(ac_fitness)
                if(is_suc==1):
                    total_success+=1
                    action_suc[int(action_type)]+=ac_fitness#action_suc最后会得到该轮统一评估中选取同样行动的产生进步的任务的fitness变化之和
                    #action_successes_num[int(action_type)]+=is_suc
             
            action_result=np.true_divide(action_suc,action_num)#得到每个action的进步的fitness之和比上该轮评估中当前action一共采取的次数，用以更改β分布的值
            #action_successes_rate=np.true_divide(action_successes_num,action_num)
            action_choose=np.argmax(action_result)
            if len(to_evaluate1)!=0:
             per_toevaluate1=b_evals/ params['dump_period']
             per_success=total_success/b_evals
            # if np.argmax(action_successes_rate)>0.1:
            #     action_choose=np.argmax(action_result) 
            # else:
            #     action_choose=np.random.randint(0, 2,dtype='int')
            for act in range(0,10):#选取表现最好的任务增加其再次被选取的概率
                if act==action_choose:
                    action[act][0]+=0.25
                else:
                    action[act][1]+=0.25
            # print(action_suc)
            # print('本轮选择各种策略的个数为')
            # print(action_num)
            # print(action_result)
            # print('当前选定增加概率的动作为')
            # print(action_choose)
            # print('各種動作的成功率為')
            #print(action_successes_rate)
            n_e = [len(v) for v in successes.values()]
            print(n_evals, n_e)
            print(action_choose)
            np.savetxt('t_size.dat', np.array(n_e))
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {}\n".format(n_evals, len(archive.keys()), fit_list.max(), fit_list.mean()))
            log_file.flush()
    cm.__save_archive(archive, n_evals)
    return archive

# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])
    # CVT-based version
    my_map = compute(dim_map=2, dim_x = 10, n_niches=1500, f=rastrigin)
