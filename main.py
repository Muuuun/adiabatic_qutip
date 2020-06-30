import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from scipy.optimize import minimize
import cmath

def rotate_right(l,n):
    return l[-n:]+l[:-n]

def individual_detection(spin_number, phonon_number, i):
    obserable_element = [(sigmaz()+qeye(2))/2]+[qeye(2) for j in np.arange(spin_number-1)]
    return tensor(rotate_right(obserable_element,i)+[qeye(phonon_number)])

def global_rotation(spin_number):
    spin_part = [(1+1j*sigmax())/np.sqrt(2) for i in np.arange(spin_number)]
    return tensor(spin_part)

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def H_LaserIon_pattern_multimodes(spin_number, laser_field, pattern, cut_off, eta, nu):
    spin_part = [sigmap()]+[qeye(2) for i in np.arange(spin_number-1)]
    motion_part = [create(cut_off)*destroy(cut_off)]+[qeye(cut_off) for i in np.arange(spin_number - 1)]
    H_inter = sum([tensor(rotate_right(spin_part,i)+[(-1j*eta*pattern[j][i]*(create(cut_off)+destroy(cut_off))).expm() for j in np.arange(spin_number)]) for i in np.arange(spin_number)])
    H_nu = sum([nu[i]*tensor([qeye(2) for i in np.arange(spin_number)]+rotate_right(motion_part,i)) for i in np.arange(spin_number)])
    H_inlist = [[H_inter, i[0]] for i in laser_field]
    H_inlistdag = [[H_inter.dag(), i[1]] for i in laser_field]
    H = [H_nu]+H_inlist+H_inlistdag
    return H

def spin_phonon_entanglement(state):
    eigen = np.abs(np.linalg.eig(state)[0])
    return -sum([value*np.log(value) for value in eigen])
    
  
def coefficient_adiabatic_P(t,arg):   
    return (arg['amplitude_BSB']*cmath.exp(t*arg['detunning_BSB']*1j+arg['phaseBSB']*1j)+arg['amplitude_RSB']*cmath.exp(t*arg['detunning_RSB']*1j+arg['phaseRSB']*1j))*(1-np.exp(-t/arg['scale']))

def coefficient_adiabatic_N(t,arg):   
    return (arg['amplitude_BSB']*cmath.exp(-t*arg['detunning_BSB']*1j-arg['phaseBSB']*1j)+arg['amplitude_RSB']*cmath.exp(-t*arg['detunning_RSB']*1j-arg['phaseRSB']*1j))*(1-np.exp(-t/arg['scale']))

def coefficient_car_P(t,arg):   
    return arg['amplitude_car']*cmath.exp(t*arg['detunning_car']*1j)*np.exp(-t/arg['scale'])

def coefficient_car_N(t,arg):   
    return arg['amplitude_car']*cmath.exp(-t*arg['detunning_car']*1j)*np.exp(-t/arg['scale'])

spin_number = 3
nu = [1.33*2*np.pi, 1.27*2*np.pi, 1.22*2*np.pi]
pattern = np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[1/np.sqrt(2),0,-1/np.sqrt(2)],[0.408,-0.816,0.408]])
eta = 0.1

scale = 70
amplitude_car = np.pi/20
amplitude_BSB = np.pi/20
amplitude_RSB = np.pi/20

detunning_BSB = nu[0] - 0.1
detunning_RSB = -nu[0] + 0.1
detunning_car = 0
phase = 0
cut_off = 5
duration = 500
laser_field = [[coefficient_adiabatic_P, coefficient_adiabatic_N],[coefficient_car_P, coefficient_car_N]]

c_op_list = []
tlist = np.linspace(0,duration,500)
psi0=tensor([(basis(2,1)-basis(2,0))/np.sqrt(2) for i in np.arange(spin_number)] + [basis(cut_off,0) for i in np.arange(spin_number)])

ARGS = {'amplitude_BSB':amplitude_BSB, 'amplitude_RSB':amplitude_RSB, \
        'detunning_BSB':detunning_BSB, 'detunning_RSB':detunning_RSB, \
        'phaseBSB':0, 'phaseRSB':0, 'amplitude_car': amplitude_car, 'detunning_car': detunning_car, 'scale': scale};

output1 = mesolve(H_LaserIon_pattern_multimodes(spin_number, laser_field, pattern, cut_off, eta, nu), psi0, tlist, c_op_list, [], args = ARGS, options=Options(store_states=True));
states = [global_rotation(spin_number)*(output1.states[i].ptrace(np.arange(spin_number)))*global_rotation(spin_number).dag() for i in np.arange(len(output1.states))]


figsize = (24, 8)
cols = 4
rows = 2**spin_number//4
fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
axs = trim_axs(axs, 2**spin_number)
for ax, i in zip(axs, np.arange(2**spin_number)):
    ax.set_title('spin state=%s' %  bin(2**spin_number - 1 - i), fontsize=20)
    ax.plot(tlist, [np.real(states[j][i,i]) for j in np.arange(len(states))], 'r')
    
    
    
plt.plot(tlist, [spin_phonon_entanglement(state) for state in states])
