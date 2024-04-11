
# either create an automata object or import built-in ones
from src.automata.minecraft_1 import minecraft_1
from src.automata.minecraft_2 import minecraft_2
from src.automata.minecraft_6 import minecraft_6
from src.automata.minecraft_7 import minecraft_7
# either create an environment object or import built-in ones
from src.environments.minecraft import minecraft
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import random

from matplotlib import colors
from datetime import datetime
import dill

import warnings

warnings.filterwarnings("ignore")


def corridor_plotter(numpy_array_q):
    """
    This function plots the confidence interval of the value function at the initial state given
    the numpy array of the value function at the initial state for each corridor.
    """
    mean = np.mean(numpy_array_q, axis=0)
    std = np.std(numpy_array_q, axis=0)
    plt.plot(range(len(mean)), mean, color='darkblue')
    plt.fill_between(range(len(mean)), mean - std, mean + std, color='royalblue', alpha=0.2)
    plt.xlabel('episode')
    plt.ylabel('value function at s_0')
    plt.title('value function at s_0 for each episode')
    plt.show()


def variable_saver():
    """
    This function uses dill to save all the variables in the workspace.
    """
    # get all the variables in the workspace
    variables = globals()
    # save the variables
    with open('meta_ec_meta.pkl', 'wb') as f:
        dill.dump(variables, f)


def reward(reward_flag):
    if reward_flag > 0:
        return .1
    elif reward_flag == -1:
        return -.1
    else:
        return 0


algorithm = 'ql'
number_of_episodes = 5000
iteration_threshold = 1000
MDP = minecraft
DFA = minecraft_1(accepting_set=[0])  # base FSMs
DFA2 = minecraft_2(accepting_set=[0, 1])  # safeguard 1
DFA3 = minecraft_6(accepting_set=[0, 1, 2])  # safeguard 2
DFA4 = minecraft_7(accepting_set=[0, 1, 2, 3, 4])  # safeguard 3
DFAs = [DFA, DFA2, DFA3, DFA4]
discount_factor = 0.9
learning_rate = 0.9
decaying_learning_rate = False
epsilon = 0.1
save_dir = './results'
test = False
average_window = -1

gamma = discount_factor
alpha = learning_rate
decay_lr = decaying_learning_rate
if decay_lr:
    alpha_initial_value = learning_rate
    alpha_final_value = 0.1
epsilon = epsilon
# path_length = []

episode = 0
DFA_indx = int(episode/1666)

MDP.reset()
DFA.reset()
DFA2.reset()
DFA3.reset()
DFA4.reset()
Q_initial_value = 0
Q = {}
q_at_initial_state = []
U_initial_value = 0
U = {}
u_at_initial_state = []

if DFA.accepting_set is None:
    raise Exception('LDBA object is not defined properly. Please specify the "accepting_set".')

# product MDP: synchronise the MDP state with the automaton state
current_state = MDP.current_state + [DFAs[DFA_indx].automaton_state]
product_MDP_action_space = MDP.action_space

# initialise Q-value outside the main loop
Q[str(current_state)] = {}
for action_index in range(len(product_MDP_action_space)):
    Q[str(current_state)][product_MDP_action_space[action_index]] = Q_initial_value
U[str(current_state)] = {}
for action_index in range(len(product_MDP_action_space)):
    U[str(current_state)][product_MDP_action_space[action_index]] = U_initial_value

# main loop
try:
    episode = 0
    path_length = [float("inf")]
    safety_violation_counter = np.zeros((1, number_of_episodes))
    policy_cover = np.zeros((1, number_of_episodes))
    while episode < number_of_episodes:
        DFA_indx = int(episode/1666)
        episode += 1
        MDP.reset()
        DFAs[DFA_indx].reset()
        current_state = MDP.current_state + [DFAs[DFA_indx].automaton_state]

        # Q value at the initial state
        Q_at_initial_state = []
        for action_index in range(len(product_MDP_action_space)):
            Q_at_initial_state.append(Q[str(current_state)][product_MDP_action_space[action_index]])
        # value function at the initial state
        q_at_initial_state.append(max(Q_at_initial_state))
        # U value at the initial state
        U_at_initial_state = []
        for action_index in range(len(product_MDP_action_space)):
            U_at_initial_state.append(U[str(current_state)][product_MDP_action_space[action_index]])
        # value function at the initial state
        u_at_initial_state.append(min(U_at_initial_state))
        print('episode:' + str(episode)
              + ', value function at s_0=' + str(max(Q_at_initial_state))
              # + ', trace length=' + str(path_length[len(path_length) - 1])
              # + ', learning rate=' + str(alpha)
              # + ', current state=' + str(MDP.current_state)
              )
        iteration = 0
        path = current_state

        # annealing the learning rate
        if decay_lr:
            alpha = max(alpha_final_value,
                        ((alpha_final_value - alpha_initial_value) / (0.8 * number_of_episodes))
                        * episode + alpha_initial_value)

        # each episode loop
        # while DFA.accepting_set and \
        while iteration < iteration_threshold and \
                DFAs[DFA_indx].automaton_state != -1:
            iteration += 1

            # find the action with max Q at the current state
            Qs = []
            for action_index in range(len(product_MDP_action_space)):
                Qs.append(Q[str(current_state)][product_MDP_action_space[action_index]])
            maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
            maxQ_action = product_MDP_action_space[maxQ_action_index]
            Us = []
            for action_index_u in range(len(product_MDP_action_space)):
                Us.append(U[str(current_state)][product_MDP_action_space[action_index_u]])
            maxU_action_index = random.choice(np.where(Us == np.min(Us))[0])
            maxU_action = product_MDP_action_space[maxU_action_index]

            # product MDP modification (for more details refer to the tool paper)
            # epsilon-greedy policy
            if random.random() < epsilon:
                next_MDP_state = MDP.step(random.choice(MDP.action_space))
            else:
                next_MDP_state = MDP.step(maxQ_action)
            next_automaton_state = DFAs[DFA_indx].step(MDP.state_label(next_MDP_state))

            # product MDP: synchronise the automaton with MDP
            next_state = next_MDP_state + [next_automaton_state]

            # Q values of the next state
            Qs_prime = []
            if str(next_state) not in Q.keys():
                Q[str(next_state)] = {}
                for action_index in range(len(product_MDP_action_space)):
                    Q[str(next_state)][product_MDP_action_space[action_index]] = Q_initial_value
                    Qs_prime.append(Q_initial_value)
                # loop over all previous DFAs to check if the state is visited before
                counter = 0
                for i in range(DFA_indx):
                    if str(next_MDP_state + [i]) in Q.keys():
                        counter += 1
                        for action_index in range(len(product_MDP_action_space)):
                            Q[str(next_state)][product_MDP_action_space[action_index]] = \
                                Q[str(next_state)][product_MDP_action_space[action_index]] + \
                                Q[str(next_MDP_state + [i])][product_MDP_action_space[action_index]]
                if counter > 0:
                    for action_index in range(len(product_MDP_action_space)):
                        Q[str(next_state)][product_MDP_action_space[action_index]] = \
                            Q[str(next_state)][product_MDP_action_space[action_index]] / counter
                for action_index in range(len(product_MDP_action_space)):
                    Qs_prime.append(Q[str(next_state)][product_MDP_action_space[action_index]])
            else:
                for action_index in range(len(product_MDP_action_space)):
                    Qs_prime.append(Q[str(next_state)][product_MDP_action_space[action_index]])
            Us_prime = []
            if str(next_state) not in U.keys():
                U[str(next_state)] = {}
                for action_index in range(len(product_MDP_action_space)):
                    U[str(next_state)][product_MDP_action_space[action_index]] = U_initial_value
                    Us_prime.append(U_initial_value)
            else:
                for action_index in range(len(product_MDP_action_space)):
                    Us_prime.append(U[str(next_state)][product_MDP_action_space[action_index]])

            # update the accepting frontier set
            reward_flag = DFAs[DFA_indx].accepting(next_automaton_state)

            if reward_flag < 0:
                state_dep_gamma = gamma
            else:
                state_dep_gamma = gamma

            # Q update
            Q[str(current_state)][maxQ_action] = \
                (1 - alpha) * Q[str(current_state)][maxQ_action] + \
                alpha * (reward(reward_flag) + state_dep_gamma * np.max(Qs_prime))
            # U update
            U[str(current_state)][maxU_action] = \
                (1 - alpha) * U[str(current_state)][maxU_action] + \
                alpha * (reward(reward_flag) + state_dep_gamma * np.min(Us_prime))

            if test:
                print(str(maxQ_action)
                      + ' | ' + str(next_state)
                      + ' | ' + MDP.state_label(next_MDP_state)
                      + ' | ' + str(reward_flag)
                      + ' | ' + str(Q[str(current_state)][maxQ_action]))

            # update the state
            current_state = next_state.copy()
            path.append(current_state)

        if DFAs[DFA_indx].automaton_state == -1:
            safety_violation_counter[0][episode - 1] = 1
        # append the path length
        path_length.append(len(path))

except KeyboardInterrupt:
    print('\nTraining exited early.')
    try:
        is_save = input(
            'Would you like to save the training data? '
            'If so, type in "y", otherwise, interrupt with CTRL+C. ')
    except KeyboardInterrupt:
        print('\nExiting...')

    if is_save == 'y' or is_save == 'Y':
        print('Saving...')
        early_interruption = 1

average_window = int(0.03 * number_of_episodes)
plt.plot(u_at_initial_state, c="royalblue")
plt.xlabel('Episode Number')
plt.ylabel('Value Function at The Initial State')
plt.grid(True)
avg = np.convolve(q_at_initial_state, np.ones((average_window,)) / average_window, mode='valid')
plt.plot(avg, c='darkblue')
plt.show()

print('testing...')
number_of_tests = 100
successes_in_test = 0
test_DFA = DFAs[-1]
number_of_safety_violations = 0
number_of_metalearning = 0
for tt in range(number_of_tests):
    MDP.reset()
    test_DFA.reset()

    test_path = [MDP.current_state]
    iteration_num = 0
    current_state = MDP.current_state + [test_DFA.automaton_state]
    while (not (test_DFA.automaton_state == -1)) and \
            iteration_num < iteration_threshold:
        iteration_num += 1

        current_state = MDP.current_state + [test_DFA.automaton_state]

        product_MDP_action_space = MDP.action_space

        Qs = []
        if str(current_state) in Q.keys():
            for action_index in range(len(product_MDP_action_space)):
                Qs.append(Q[str(current_state)][product_MDP_action_space[action_index]])
        else:
            number_of_metalearning += 1
            if str(MDP.current_state + [0]) in Q.keys():
                for action_index in range(len(product_MDP_action_space)):
                    Qs.append(Q[str(MDP.current_state + [0])][product_MDP_action_space[action_index]])
            else:
                Qs.append(0)
        maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
        maxQ_action = product_MDP_action_space[maxQ_action_index]
        next_MDP_state = MDP.step(maxQ_action)
        next_automaton_state = test_DFA.step(MDP.state_label(next_MDP_state))

        test_path.append(next_MDP_state)

    if test_DFA.automaton_state == -1:
        number_of_safety_violations += 1
    else:
        successes_in_test += 1
print('success rate in testing: ' + str(100 * successes_in_test / number_of_tests) + '%')
print('safety violations: ' + str(100 * number_of_safety_violations / number_of_tests) + '%')
print('number of meta-learning queries: ' + str(number_of_metalearning))

# plt.plot(path_length, c='royalblue')
# plt.xlabel('Episode Number')
# plt.ylabel('Agent Traversed Distance from The Initial State')
# plt.grid(True)
# if average_window > 0:
#     avg = np.convolve(path_length, np.ones((average_window,)) / average_window, mode='valid')
#     plt.plot(avg, c='darkblue')
# plt.savefig(os.path.join(results_sub_path, 'traversed distance in the grid.png'))
# plt.show()

distinct_labels = np.unique(MDP.labels)
labels_dic = {}
label_indx = 0
bounds = [-0.9]
cmap = plt.get_cmap('gist_rainbow')
for label in distinct_labels:
    labels_dic[label] = label_indx
    bounds.append(bounds[-1] + 1)
    label_indx += 1
color_map = cmap(np.linspace(0, 1, len(distinct_labels)))
cmap = colors.ListedColormap(color_map)
norm = colors.BoundaryNorm(bounds, cmap.N)
labels_values = np.zeros([MDP.shape[0], MDP.shape[1], 100])
value_function = np.zeros([MDP.shape[0], MDP.shape[1]])
for av in range(100):
    MDP.reset()
    for i in range(MDP.shape[0]):
        for j in range(MDP.shape[1]):
            labels_values[i][j][av] = labels_dic[MDP.state_label([i, j])]
            if str([i, j, 0]) in Q.keys():
                v = []
                for action_index in range(len(product_MDP_action_space)):
                    v.append(Q[str([i, j, 0])][product_MDP_action_space[action_index]])
                value_function[i][j] = max(v)
patches = [mpatches.Patch(color=color_map[i], label=list(distinct_labels)[i]) for i in
           range(len(distinct_labels))]
# pdb.set_trace()
# labels_value = np.mode(labels_values, axis=2)
labels_value = labels_values[:, :, 99]
plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
path_x, path_y = np.array(test_path).T
plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
plt.scatter(path_y[0], path_x[0], c='red', edgecolors='black')
plt.annotate('s_0', (path_y[0], path_x[0]), fontsize=15, xytext=(20, 20), textcoords="offset points",
             va="center", ha="left",
             bbox=dict(boxstyle="round", fc="w"),
             arrowprops=dict(arrowstyle="->"))
plt.title('This policy is synthesised by the trained agent')
results_path = os.path.join(os.getcwd(), save_dir[2:])
dt_string = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
results_sub_path = os.path.join(os.getcwd(), save_dir[2:], dt_string)
if not os.path.exists(results_path):
    os.mkdir(results_path)
os.mkdir(results_sub_path)
plt.savefig(
    os.path.join(results_sub_path, 'tested_policy.png'), bbox_inches="tight")
plt.show()

variable_saver()
