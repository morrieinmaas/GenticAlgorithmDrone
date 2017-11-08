##This is my solution to som eclass room excercises
##this will not un on your machine as you will lack utils/*.py
##such as utilts/history.py ordrone.py, because they are not my code,
##but intellectual property of my professor

##The problem in hand, is concerned with optimizing a drone's flight with respect to a referene trajctory.

##The drone takes 3 parameters as an input with range 0.001 to 10000.
##Here the problem is addressed with a genetic algorithm

##You may use and reuse this code for any purpose without permiss
##ion. I hope you wi find it useful or helpful to address similar problem or
## to gain an overview on how to formulate this in python. 
 
from __future__ import print_function, division
import numpy as np 
plt.rc('figure', figsize=(8.0, 4.0), dpi=140)

import matplotlib as mpl   
import matplotlib.pyplot as plt

import utils.history; import imp; imp.reload(utils.history)
from utils.history import History


def sample_function():
    theta = np.random.uniform(0.0001,10000.0, (1,3)) # 64 random 2D points
    return theta

def obj_function(pid):
    controller_pid = drone.pid_controller((pid))
    flightpath = drone.simulate(controller_pid, reference_trajectory)
    obj = np.mean(np.sqrt((flightpath-reference_trajectory)**2))
    return obj
   

from math import sqrt

from joblib import Parallel, delayed

def make_population(size):
    population = np.random.uniform(0.0001,10000.0, (size,3))
    return population

def mutate(population, mutation_rate):
    pop_size = population.shape[0] * population.shape[1] 
    #we have to round the mutaion rate and popsize relatively, because only int values are allowed
    changes = np.round(pop_size * mutation_rate)
    mask = np.random.randint(0,2, size=population.shape).astype(np.bool)
    r = np.random.rand(*population.shape)*np.max(population)
    population[mask] = r[mask]
    return population

def selection(population, scores, cut_off):
    new_pop = np.zeros(population.shape)
    cut_off = np.round(cut_off * population.shape[0])
    #this cuts off 
    cut = np.argsort(scores)[:int(cut_off)]
    winners = population[cut,:]
    #repopulate a new population with the winners of the old one (randomly)
    #this is a shortcut, possibly biasing towards some members of the winning pop
    for i in range(0, new_pop.shape[0]):
        rand_child = np.random.randint(0, winners.shape[0])
        new_pop[i,:] = winners[rand_child,:]
    return new_pop

def cross_over(population, crossover_point):
    #this is unneccessary because our problem has 3 variables but for re-using it an diferent problems 
    #it might be handy
    cross_over_int = int(np.round(population.shape[1] * crossover_point) )
    if (0 < cross_over_int < int(population.shape[1]) ):
        for i in range(0, population.shape[0] - 2, np.round(population.shape[0])):   
            population[[i, i+1],cross_over_int:] = population[[i + 1, i],cross_over_int:]
    return population

def results_per_genereation(L, population):
    o = History()
    for i in range(0, population.shape[0]):
        theta = population[i,:]       
        o.track(theta, obj_function(theta))    
    return o.finalise()
    
def perform_GA(iters, pop_size, mutation_rate, cross, survival_rate):
    #create an initial population
    population = make_population(pop_size)
    
    results = []
    winner_of_round = np.zeros((iters, 4))
    for i in range(0,iters):
        #print(population)
        population = mutate(population, mutation_rate=mutation_rate)
        #print(population)
        results = results_per_genereation(obj_function, population)
        #print(results.loss_trace)
        results_index = results.all_loss
        
        winner_of_round[i,0] = results.best
        #print(results.best)
        winner_of_round[i, 1:] = results.best_theta
        #print(results.best_theta)
        population = selection(population=population, scores=results_index, cut_off=survival_rate)
        #print(population)
        #if cross_over is not None:
        if (0.0 < cross < 1.0):
            population = cross_over(population=population, crossover_point=cross)

    #define a winning solution
    winner_ind = np.argsort(winner_of_round[:,0])[0]
    winner = winner_of_round[winner_ind,:]
    
    #return the best solution  
    #use for running in parallel/loop and for running test trials with different parameters
    return winner

    #return the solutions of all generations from a certain parameter setting
    #this way this returns the history of the iterations for a single provided setup
    #return winner_of_round

#num_cores = multiprocessing.cpu_count()


#for the following 4 use return winner_winner_of_round
#after running this bit comment it out until definition of a1
#results_all_gen_4_fixed20_Xat1 = perform_GA(iters=20, pop_size=40, mutation_rate=0.05, cross = 0.3, survival_rate=0.25)
#results_all_gen_4_fixed20_Xat2 = perform_GA(iters=20, pop_size=40, mutation_rate=0.05, cross = 0.7, survival_rate=0.25)

#results_all_gen_4_fixed40_Xat1 = perform_GA(iters=40, pop_size=40, mutation_rate=0.05, cross = 0.3, survival_rate=0.25)
#results_all_gen_4_fixed40_Xat2 = perform_GA(iters=40, pop_size=40, mutation_rate=0.05, cross = 0.7, survival_rate=0.25)

good_results_all_gen_4_fixed40_Xat2 = perform_GA(iters=40, pop_size=40, mutation_rate=0.05, cross = 0.7, survival_rate=0.25)

a1 = [5, 10, 20, 30, 40]
b1 = [0.05, 0.1, 0.2, 0.35, 0.5] 
c1 = 0.3
c2 = 0.7

def res3d_X(a, b, cr):
    results3D = [] #np.zeros((5,5))
    for i in a:
        for j in b:
            results3D.append(perform_GA(iters=40, pop_size=i, mutation_rate=b, cross = cr, survival_rate=0.25)[0])
    return results3D

#after running the first trials chnage the return in perform_GA
# and uncomment results_4_3DX1 and results_4_3DX2

#find results in solution space population between 0 and 40 and motation rate between 5% and 50%
#cross over after 1 and 2 respectivly
#this will take a while!

#for the following 2 use return winner in peform_GA
#this also takes a while!

#results_4_3DX1 = res3d_X(a1, b1, c1)
#results_4_3DX2 = res3d_X(a1, b1, c2)


'''
#for the following use return winner_of_round
#you can loop through solutin spaces in parallel as follows:

#results_pop_i_sr25 = Parallel(n_jobs=num_cores-1)(delayed(perform_GA)(iters=40, pop_size=i,mutation_rate=0.05, cross = 0, survival_rate=0.25) for i in [5, 10, 20, 30, 40])

results_muti_pop5_sr25 = Parallel(n_jobs=num_cores-1)(delayed(perform_GA)(iters=40, pop_size=5,mutation_rate=i, cross = 0, survival_rate=0.25) for i in [0.05, 0.1, 0.2, 0.35, 0.5])
'''

from mpl_toolkits.mplot3d import Axes3D

print((results_all_gen_4_fixed40_Xat2[39,:]))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0, 41, 40), results_all_gen_4_fixed40_Xat1[:, 0])

ax.set_xlabel("iterations")
ax.set_ylabel("score of objective function")
ax.set_title("GA performance - iters 20, population 40, mutation rate 5%, crossover after col 1 and survival rate 25%" + "\n")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0, 41, 40), results_all_gen_4_fixed40_Xat2[:, 0])

ax.set_xlabel("iterations")
ax.set_ylabel("score of objective function")
ax.set_title("GA performance - iters 40, population 40, mutation rate 5%, crossover after col 2 and survival rate 25%" + "\n")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0, 21, 20), results_all_gen_4_fixed20_Xat1[:, 0])

ax.set_xlabel("iterations")
ax.set_ylabel("score of objective function")
ax.set_title("GA performance - iters 20, population 40, mutation rate 5%, crossover after col 1 and survival rate 25%" + "\n")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0, 21, 20), results_all_gen_4_fixed20_Xat2[:, 0])

ax.set_xlabel("iterations")
ax.set_ylabel("score of objective function")
ax.set_title("GA performance - iters 40, population 40, mutation rate 5%, crossover after col 2 and survival rate 25%" + "\n")


print("We see that after appr.40 the latest iterations there is no improvement in performance, \n so all following trials will be cut off at 40 iterations \n")

print("We also see, that crossover after the 2nd entry of 3 values \n flattens out the results nicely and less oscillation results")

print("Further, in the wireframe plots, it becomes clear, that the population size is more important")
print("than the mutation rate for optimizing the results")
print("In fact, a mutation rate higher than 20 % decreses the goodness of the resuts")


#copy of what i used to create wireframe1.png - these are results from my machine

results_4_3D = [1.93234329807241, 0.48422994502221162, 5.5823629674517745, 2.8313550888717831, 6.1016605801592227, 1.6713673237593871, 1.7088806661870914, 5.3509085603136741, 6.4447664054120848, 0.72878526852686754, 0.34996100891675358, 5.9840208232570777, 2.7142619665161725, 3.1621377673938453, 2.5751877269043675, 0.47429191817929151, 0.40440713532672362, 0.25734622975739502, 2.0227533657685415, 2.0214806611605969, 0.40714340714637909, 0.51697427813860197, 0.58254405220242045, 0.31027678421310834, 0.46093927364327969]


a2 = np.array([a1, a1, a1, a1, a1])
b2 = np.array([b1, b1, b1, b1, b1])

z1 = np.array(results_4_3D).reshape((5, 5))

fig2 = plt.figure()
ax_1 = fig2.add_subplot(111, projection='3d')
ax_1.plot_wireframe(a2.T, b2, z1)

ax_1.set_xlabel("Population size")
ax_1.set_ylabel("Mutation rate")
ax_1.set_zlabel("Score of loss function")
ax_1.set_title("Performance of GA for different meta-heuristics \n")

#uncomment the following after finding results_4_3DX1 and results_4_3DX2
'''
z2 = np.array(results_4_3DX1)[:,0].reshape((5, 5))

fig2 = plt.figure()
ax_1 = fig2.add_subplot(111, projection='3d')
ax_1.plot_wireframe(a2.T, b2, z2)

ax_1.set_xlabel("Population size")
ax_1.set_ylabel("Mutation rate")
ax_1.set_zlabel("Score of loss function")
ax_1.set_title("Performance of GA for different meta-heuristics \n")

#this is actually the best/clearest wireframe plot with these settings
z3 = np.array(results_4_3DX2)[:,0].reshape((5, 5))

fig2 = plt.figure()
ax_1 = fig2.add_subplot(111, projection='3d')
ax_1.plot_wireframe(a2.T, b2, z3)

ax_1.set_xlabel("Population size")
ax_1.set_ylabel("Mutation rate")
ax_1.set_zlabel("Score of loss function")
ax_1.set_title("Performance of GA for different meta-heuristics \n")
'''

