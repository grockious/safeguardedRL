import numpy as np
import random


class RLcore:
    """
    LCRL

    ...

    Attributes
    ----------
    MDP : an object with the following properties
        (1) a "step(action)" function describing the dynamics
        (2) a "reset()" function that resets the state to an initial state
        (3) a "state_label(state)" function that maps states to labels
        (4) current state of the MDP is "current_state"
        (5) action space is "action_space" and all actions are enabled in each state
    DFA : an object of ./src/automata/LDBA.py
        a limit-deterministic Büchi automaton
    discount_factor: float
        discount factor (default 0.9)
    learning_rate: float
        learning rate (default 0.9)
    epsilon: float
        tuning parameter for the epsilon-greedy exploration scheme (default 0.1)

    Methods
    -------
    train_ql(number_of_episodes, iteration_threshold, Q_initial_value)
        employs Episodic Q-learning to synthesise an optimal policy over the product MDP
    train_nfq(number_of_episodes, iteration_threshold, nfq_replay_buffer_size, num_of_hidden_neurons)
        employs Episodic Neural Fitted Q-Iteration to synthesise an optimal policy over the product MDP
    train_ddpg(number_of_episodes, iteration_threshold, ddpg_replay_buffer_size, num_of_hidden_neurons)
        employs Episodic Deep Deterministic Policy Gradient to synthesise an optimal policy over the product MDP
    reward(automaton_state)
        shapes the reward function according to the automaton accepting set
        for more details refer to the tool paper
    action_space_augmentation()
        augments the action space whenever an epsilon transition is expected
        for more details refer to the tool paper
    """

    def __init__(
            self, MDP=None,
            DFA=None,
            discount_factor=0.9,
            learning_rate=0.9,
            decaying_learning_rate=True,
            epsilon=0.15
    ):
        if MDP is None:
            raise Exception("LCRL expects an MDP object as input")
        self.MDP = MDP
        if DFA is None:
            raise Exception("LCRL expects an LDBA object as input")
        self.DFA = DFA
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.decay_lr = decaying_learning_rate
        if self.decay_lr:
            self.alpha_initial_value = learning_rate
            self.alpha_final_value = 0.1
        self.epsilon = epsilon
        self.path_length = []
        self.Q = {}
        self.replay_buffers = {}
        self.Q_initial_value = 0
        self.early_interruption = 0
        self.q_at_initial_state = []
        self.successes_in_test = 0
        # ##### testing area ##### #
        self.test = False
        # ######################## #

    def train_ql(
            self, number_of_episodes,
            iteration_threshold,
            Q_initial_value=0
    ):
        self.MDP.reset()
        self.DFA.reset()
        self.Q_initial_value = Q_initial_value

        if self.DFA.accepting_set is None:
            raise Exception('LDBA object is not defined properly. Please specify the "accepting_set".')

        # product MDP: synchronise the MDP state with the automaton state
        current_state = self.MDP.current_state + [self.DFA.automaton_state]
        product_MDP_action_space = self.MDP.action_space

        # initialise Q-value outside the main loop
        self.Q[str(current_state)] = {}
        for action_index in range(len(product_MDP_action_space)):
            self.Q[str(current_state)][product_MDP_action_space[action_index]] = Q_initial_value

        # main loop
        try:
            episode = 0
            self.path_length = [float("inf")]
            while episode < number_of_episodes:
                episode += 1
                self.MDP.reset()
                self.DFA.reset()
                current_state = self.MDP.current_state + [self.DFA.automaton_state]

                # Q value at the initial state
                Q_at_initial_state = []
                for action_index in range(len(product_MDP_action_space)):
                    Q_at_initial_state.append(self.Q[str(current_state)][product_MDP_action_space[action_index]])
                # value function at the initial state
                self.q_at_initial_state.append(max(Q_at_initial_state))
                print('episode:' + str(episode)
                      + ', value function at s_0=' + str(max(Q_at_initial_state))
                      # + ', trace length=' + str(self.path_length[len(self.path_length) - 1])
                      # + ', learning rate=' + str(self.alpha)
                      # + ', current state=' + str(self.MDP.current_state)
                      )
                iteration = 0
                path = current_state

                # annealing the learning rate
                if self.decay_lr:
                    self.alpha = max(self.alpha_final_value,
                                     ((self.alpha_final_value - self.alpha_initial_value) / (0.8 * number_of_episodes))
                                     * episode + self.alpha_initial_value)

                # each episode loop
                # while self.DFA.accepting_set and \
                while (not self.DFA.accepting(self.DFA.automaton_state)) and iteration < iteration_threshold and \
                        self.DFA.automaton_state != -1:
                    iteration += 1

                    # find the action with max Q at the current state
                    Qs = []
                    for action_index in range(len(product_MDP_action_space)):
                        Qs.append(self.Q[str(current_state)][product_MDP_action_space[action_index]])
                    maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
                    maxQ_action = product_MDP_action_space[maxQ_action_index]

                    # product MDP modification (for more details refer to the tool paper)
                    # epsilon-greedy policy
                    if random.random() < self.epsilon:
                        next_MDP_state = self.MDP.step(random.choice(self.MDP.action_space))
                    else:
                        next_MDP_state = self.MDP.step(maxQ_action)
                    next_automaton_state = self.DFA.step(self.MDP.state_label(next_MDP_state))

                    # product MDP: synchronise the automaton with MDP
                    next_state = next_MDP_state + [next_automaton_state]

                    # Q values of the next state
                    Qs_prime = []
                    if str(next_state) not in self.Q.keys():
                        self.Q[str(next_state)] = {}
                        for action_index in range(len(product_MDP_action_space)):
                            self.Q[str(next_state)][product_MDP_action_space[action_index]] = Q_initial_value
                            Qs_prime.append(Q_initial_value)
                    else:
                        for action_index in range(len(product_MDP_action_space)):
                            Qs_prime.append(self.Q[str(next_state)][product_MDP_action_space[action_index]])

                    # update the accepting frontier set
                    reward_flag = self.DFA.accepting(next_automaton_state)

                    if reward_flag > 0:
                        state_dep_gamma = self.gamma
                    else:
                        state_dep_gamma = 1

                    # Q update
                    self.Q[str(current_state)][maxQ_action] = \
                        (1 - self.alpha) * self.Q[str(current_state)][maxQ_action] + \
                        self.alpha * (self.reward(reward_flag) + state_dep_gamma * np.max(Qs_prime))

                    if self.test:
                        print(str(maxQ_action)
                              + ' | ' + str(next_state)
                              + ' | ' + self.MDP.state_label(next_MDP_state)
                              + ' | ' + str(reward_flag)
                              + ' | ' + str(self.Q[str(current_state)][maxQ_action]))

                    # update the state
                    current_state = next_state.copy()
                    path.append(current_state)

                # append the path length
                self.path_length.append(len(path))

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
                self.early_interruption = 1

    def train_pi(
            self, number_of_episodes,
            iteration_threshold,
            Q_initial_value=0
    ):
        self.MDP.reset()
        self.DFA.reset()
        # initialize the policy (uniform over actions)
        action_space_cardi = len(self.MDP.action_space)
        policy = (1 / action_space_cardi) * np.ones((self.MDP.shape[0], self.MDP.shape[1], action_space_cardi))
        # initialize the value function (uniform over states)
        self.Q_initial_value = Q_initial_value
        value = Q_initial_value * np.ones((self.MDP.shape[0], self.MDP.shape[1]))

        # main loop
        try:
            episode = 0
            self.path_length = [float("inf")]
            while episode < number_of_episodes:
                episode += 1
                
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
                self.early_interruption = 1

        def policy_eval(env, val, pol, theta, gamma=self.gamma):
            delta = theta + 1
            while delta >= theta:
                old_val = val.copy()
                delta = 0

                # Traverse all states
                for x in range(env.n):
                    for y in range(env.n):
                        # Run one iteration of the Bellman update rule for the value function
                        bellman_update(env, val, old_val, x, y, pol, gamma)
                        # Compute difference
                        delta = max(delta, abs(old_val[x, y] - val[x, y]))

        def bellman_update(env, v, old_v, x, y, pi, gamma):
            # The value function on the terminal state always has value 0
            if x == env.e_x and y == env.e_y:
                return None

            total = 0

            for a in env.actions:
                # Get next state
                s_prime_x, s_prime_y = get_next_state(x, y, a, env.n)

                total += pi[x, y, a] * (env.rewards[s_prime_x, s_prime_y] + gamma * old_v[s_prime_x, s_prime_y])

            # Update the value function
            v[x, y] = total


    def train_nfq(
            self, number_of_episodes,
            iteration_threshold,
            nfq_replay_buffer_size,
            num_of_hidden_neurons=256
    ):
        import tensorflow as tf
        from tensorflow import keras
        tf.get_logger().setLevel('ERROR')
        self.MDP.reset()
        self.DFA.reset()

        if self.DFA.accepting_set is None:
            raise Exception('LDBA object is not defined properly. Please specify the "accepting_set". ')

        # ## exploration phase ## #
        # nfq_modules
        self.Q = {}
        self.replay_buffers = {}
        state_dimension = len(self.MDP.current_state)
        action_dimension = 1
        nfq_input_dim = state_dimension + action_dimension
        product_MDP_action_space = self.MDP.action_space

        # initiate an NFQ module
        model_0 = keras.Sequential([
            keras.layers.Dense(num_of_hidden_neurons, input_dim=nfq_input_dim, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.leaky_relu)])
        model_0.compile(loss='mean_squared_error',
                        metrics=['mean_squared_error'],
                        optimizer='Adam')
        self.Q[self.DFA.automaton_state] = model_0
        self.replay_buffers[self.DFA.automaton_state] = []

        try:
            # exploration loop
            episode = 0
            print("sampling...")
            rewarding_paths = []
            while episode < number_of_episodes:
                episode_path = []
                episode += 1
                self.MDP.reset()
                self.DFA.reset()
                current_state = self.MDP.current_state.tolist() + [self.DFA.automaton_state]

                iteration = 0

                if self.decay_lr:
                    try:
                        input(
                            'Decaying learning rate has to be set to False in "nfq". '
                            'Would you like me to set it to False and continue? '
                            'If so, type in "y", otherwise, interrupt with CTRL+C. ')
                    except KeyboardInterrupt:
                        print('\nExiting...')

                # each episode loop
                while self.DFA.accepting_set and \
                        iteration < iteration_threshold and \
                        self.DFA.automaton_state != -1:
                    iteration += 1

                    # find the action with max Q at the current state
                    active_model = self.DFA.automaton_state

                    action_index = random.randint(0, len(product_MDP_action_space) - 1)
                    maxQ_action = product_MDP_action_space[action_index]

                    next_MDP_state = self.MDP.step(maxQ_action).tolist()
                    next_automaton_state = self.DFA.step(self.MDP.state_label(next_MDP_state))

                    # product MDP: synchronise the automaton with MDP
                    next_state = next_MDP_state + [next_automaton_state]

                    if next_automaton_state == -1:
                        break

                    if self.test:
                        print(str(maxQ_action) + ' | ' + str(next_state) + ' | ' + self.MDP.state_label(next_MDP_state))

                    reward_flag = self.DFA.accepting(next_automaton_state)
                    reward_flag = reward_flag + (
                        np.cos(np.pi * reward_flag)) * 0.1 * random.random()  # to break symmetry

                    if reward_flag > 0.5:
                        state_dep_gamma = self.gamma
                        # print('reward reached!')
                    else:
                        state_dep_gamma = self.gamma

                    # create an NFQ module if needed
                    if next_automaton_state not in self.Q.keys() and self.DFA.accepting_set:
                        # initiate an NFQ module
                        self.Q[next_automaton_state] = \
                            keras.Sequential([
                                keras.layers.Dense(num_of_hidden_neurons, input_dim=nfq_input_dim,
                                                   activation=tf.nn.relu),
                                keras.layers.Dense(1, activation=tf.nn.leaky_relu)])
                        self.Q[next_automaton_state].compile(loss='mean_squared_error',
                                                             metrics=['mean_squared_error'],
                                                             optimizer='Adam')
                        self.replay_buffers[next_automaton_state] = []

                    # save the experience
                    # (state, action, state, reward, state_dependent_gamma, epsilon_transition_taken)
                    # to the replay buffer
                    self.replay_buffers[active_model].append(
                        current_state[0: -1] +
                        [action_index] +
                        next_state[0: -1] +
                        [reward_flag] +
                        [state_dep_gamma] +
                        [current_state[-1]]
                    )
                    episode_path.append(self.replay_buffers[active_model][-1])
                    if reward_flag > 0.5:
                        rewarding_paths.extend(episode_path)

                    # update the state
                    current_state = next_state

            # exploitation loop
            exp_size = nfq_replay_buffer_size
            high_reward_total = np.array(rewarding_paths)
            sars = {}
            high_reward_sars = {}
            for model_key in self.Q.keys():
                high_reward_sars[model_key] = high_reward_total[high_reward_total[:, -1] == model_key]
                if len(high_reward_sars[model_key]) < exp_size:
                    sars[model_key] = np.array(self.replay_buffers[model_key])
                    uniform_sampled_sars = sars[model_key][
                        np.random.choice(sars[model_key].shape[0], exp_size - len(high_reward_sars[model_key]),
                                         replace=False)]
                    sars[model_key] = np.vstack((high_reward_sars[model_key], uniform_sampled_sars))
                    np.random.shuffle(sars[model_key])
                else:
                    sars[model_key] = high_reward_sars[model_key]

            for model_key in self.Q.keys():
                self.Q[model_key].fit(
                    sars[model_key][:, 0:state_dimension + 1],  # stat - action
                    sars[model_key][:, nfq_input_dim + state_dimension],  # reward
                    epochs=3, verbose=0)

            episode = 0
            self.MDP.reset()
            self.DFA.reset()
            init_state = self.MDP.current_state.tolist().copy()
            while episode < number_of_episodes:
                # Q value at the initial state
                Q_at_initial_state = []
                for action_index in range(len(product_MDP_action_space)):
                    Q_at_initial_state.append(
                        list(self.Q[0].predict(np.array(init_state + [action_index]).reshape(1, nfq_input_dim))[0])[0]
                    )

                self.q_at_initial_state.append(max(Q_at_initial_state))
                print('episode:' + str(episode)
                      + ', value function at s_0=' + str(self.q_at_initial_state[episode])
                      # + ', trace length=' + str(self.path_length[len(self.path_length) - 1])
                      # + ', learning rate=' + str(self.alpha)
                      # + ', current state=' + str(self.MDP.current_state)
                      )

                for model_key in self.Q.keys():
                    target = np.zeros(len(sars[model_key]))
                    for k in range(len(sars[model_key])):
                        neigh = []
                        if sars[model_key][k, -1] == 1:  # meaning that an epsilon transition was active
                            action_space = list(range(len(self.MDP.action_space)))
                            action_space = action_space + [action_space[-1] + 1]  # for the epsilon transition
                        else:
                            action_space = self.MDP.action_space
                        neigh_input = []
                        for a in range(len(action_space)):
                            neigh_input = np.append(sars[model_key][k, nfq_input_dim:nfq_input_dim + state_dimension],
                                                    a).reshape(1, nfq_input_dim)
                            neigh.append(self.Q[model_key].predict(neigh_input))
                        # target = reward + gamma * max Q(s',a')
                        target[k] = sars[model_key][k, nfq_input_dim + state_dimension] + \
                                    sars[model_key][k, nfq_input_dim + state_dimension + 1] * max(neigh)
                        # terminal state check
                        if sars[model_key][k, nfq_input_dim + state_dimension] > 0.5:
                            target[k] = sars[model_key][k, nfq_input_dim + state_dimension]
                    self.Q[model_key].fit(
                        sars[model_key][:, 0:state_dimension + 1],
                        target.reshape(len(sars[model_key]), 1),
                        epochs=3,
                        verbose=0)

                episode += 1

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
                self.early_interruption = 1

    def train_ddpg(
            self, number_of_episodes,
            iteration_threshold,
            ddpg_replay_buffer_size,
            num_of_hidden_neurons=32
    ):
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from keras import layers
        state_dimension = len(self.MDP.current_state)
        action_dimension = 1
        lower_bound = -1.0
        upper_bound = 1.0

        if self.DFA.accepting_set is None:
            raise Exception('LDBA object is not defined properly. Please specify the "accepting_set". ')

        class OUActionNoise:
            def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
                self.theta = theta
                self.mean = mean
                self.std_dev = std_deviation
                self.dt = dt
                self.x_initial = x_initial
                self.reset()

            def __call__(self):
                x = (
                        self.x_prev
                        + self.theta * (self.mean - self.x_prev) * self.dt
                        + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
                )
                self.x_prev = x
                return x

            def reset(self):
                if self.x_initial is not None:
                    self.x_prev = self.x_initial
                else:
                    self.x_prev = np.zeros_like(self.mean)

        class Buffer:
            def __init__(self, buffer_capacity=ddpg_replay_buffer_size, batch_size=64, gamma=self.gamma):
                # maximum number of experience samples
                self.buffer_capacity = buffer_capacity
                self.batch_size = batch_size
                self.gamma = gamma

                # num of times record() was called.
                self.buffer_counter = 0

                self.state_buffer = np.zeros((self.buffer_capacity, state_dimension))
                self.action_buffer = np.zeros((self.buffer_capacity, action_dimension))
                self.reward_buffer = np.zeros((self.buffer_capacity, 1))
                self.next_state_buffer = np.zeros((self.buffer_capacity, state_dimension))

            # (s,a,r,s') tuple as input
            def record(self, obs_tuple):
                # buffer capacity index
                index = self.buffer_counter % self.buffer_capacity

                self.state_buffer[index] = obs_tuple[0]
                self.action_buffer[index] = obs_tuple[1]
                self.reward_buffer[index] = obs_tuple[2]
                self.next_state_buffer[index] = obs_tuple[3]

                self.buffer_counter += 1

            @tf.function
            def update(
                    self, state_batch, action_batch, reward_batch, next_state_batch,
            ):
                # training actor and critic
                with tf.GradientTape() as tape:
                    target_actions = target_actor_dict[active_model](next_state_batch, training=True)
                    y = reward_batch + self.gamma * target_critic_dict[active_model](
                        [next_state_batch, target_actions], training=True
                    )
                    critic_value = critic_dict[active_model]([state_batch, action_batch], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

                critic_grad = tape.gradient(critic_loss, critic_dict[active_model].trainable_variables)
                critic_optimizer.apply_gradients(
                    zip(critic_grad, critic_dict[active_model].trainable_variables)
                )

                with tf.GradientTape() as tape:
                    actions = actor_dict[active_model](state_batch, training=True)
                    critic_value = critic_dict[active_model]([state_batch, actions], training=True)
                    actor_loss = -tf.math.reduce_mean(critic_value)

                actor_grad = tape.gradient(actor_loss, actor_dict[active_model].trainable_variables)
                actor_optimizer.apply_gradients(
                    zip(actor_grad, actor_dict[active_model].trainable_variables)
                )

            # compute the loss and update parameters
            def learn(self):
                # sampling
                record_range = min(self.buffer_counter, self.buffer_capacity)
                batch_indices = np.random.choice(record_range, self.batch_size)

                state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
                action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
                reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
                reward_batch = tf.cast(reward_batch, dtype=tf.float32)
                next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

                self.update(state_batch, action_batch, reward_batch, next_state_batch)

        # soft update
        @tf.function
        def update_target(target_weights, weights, tau):
            for (a, b) in zip(target_weights, weights):
                a.assign(b * tau + a * (1 - tau))

        def get_actor():
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

            inputs = layers.Input(shape=(state_dimension,))
            out = layers.Dense(num_of_hidden_neurons * 8, activation="relu")(inputs)
            out = layers.Dense(num_of_hidden_neurons * 8, activation="relu")(out)
            outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic():
            # state as input
            state_input = layers.Input(shape=(state_dimension))
            state_out = layers.Dense(int(num_of_hidden_neurons / 2), activation="relu")(state_input)
            state_out = layers.Dense(num_of_hidden_neurons, activation="relu")(state_out)

            # action as input
            action_input = layers.Input(shape=(action_dimension))
            action_out = layers.Dense(num_of_hidden_neurons, activation="relu")(action_input)

            # concatenating
            concat = layers.Concatenate()([state_out, action_out])

            out = layers.Dense(num_of_hidden_neurons * 8, activation="relu")(concat)
            out = layers.Dense(num_of_hidden_neurons * 8, activation="relu")(out)
            outputs = layers.Dense(1)(out)

            # output
            model = tf.keras.Model([state_input, action_input], outputs)

            return model

        def policy(state, noise_object):
            sampled_actions = tf.squeeze(actor_dict[active_model](state))
            noise = noise_object()
            # Adding noise to action
            sampled_actions = sampled_actions.numpy() + noise

            # We make sure action is within bounds
            legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

            return [np.squeeze(legal_action)]

        exploration_parameter = self.epsilon * 3
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(exploration_parameter) * np.ones(1))

        # initiate a DDPG module
        actor_model_0 = get_actor()
        actor_dict = {0: actor_model_0}
        critic_model_0 = get_critic()
        critic_dict = {0: critic_model_0}

        target_actor_0 = get_actor()
        target_actor_dict = {0: target_actor_0}
        target_critic_0 = get_critic()
        target_critic_dict = {0: target_critic_0}

        target_actor_dict[0].set_weights(actor_dict[0].get_weights())
        target_critic_dict[0].set_weights(critic_dict[0].get_weights())

        # learning rates
        critic_lr = self.alpha * 0.04
        actor_lr = self.alpha * 0.02

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        # soft update parameter
        tau = 0.005

        buffer_0 = Buffer(ddpg_replay_buffer_size, 128)
        buffer_dict = {0: buffer_0}

        epsilon_transition_taken = 0
        ep_reward_list = []

        # training loop
        try:
            for episode in range(number_of_episodes):

                self.MDP.reset()
                self.DFA.reset()
                prev_state = self.MDP.current_state
                episodic_reward = 0

                if self.decay_lr:
                    try:
                        input(
                            'Decaying learning rate has to be set to False in "ddpg". '
                            'Would you like me to set it to False and continue? '
                            'If so, type in "y", otherwise, interrupt with CTRL+C. ')
                    except KeyboardInterrupt:
                        print('\nExiting...')

                # each episode loop
                while True:

                    # get an action from the actor
                    active_model = self.DFA.automaton_state

                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                    action = policy(tf_prev_state, ou_noise)

                    # product MDP modification (for more details refer to the tool paper)
                    state = self.MDP.step(action)
                    next_automaton_state = self.DFA.step(self.MDP.state_label(state))

                    # product MDP: synchronise the automaton with MDP
                    next_MDP_state = self.MDP.current_state.tolist()
                    next_state = next_MDP_state + [next_automaton_state]

                    if self.test:
                        print(str(action) + ' | ' + str(next_state) + ' | ' + self.MDP.state_label(next_MDP_state))

                    # update the accepting frontier set
                    reward_flag = self.DFA.accepting(next_automaton_state)

                    if reward_flag > 0.5:
                        state_dep_gamma = self.gamma
                    else:
                        state_dep_gamma = 1

                    # create a DDPG module if needed
                    if next_automaton_state not in actor_dict.keys() and self.DFA.accepting_set:
                        # initiate a DDPG module
                        actor_dict[next_automaton_state] = get_actor()
                        critic_dict[next_automaton_state] = get_critic()
                        target_actor_dict[next_automaton_state] = get_actor()
                        target_critic_dict[next_automaton_state] = get_critic()

                        buffer_dict[next_automaton_state] = Buffer(ddpg_replay_buffer_size, 128)

                        # Making the weights equal initially
                        target_actor_dict[next_automaton_state].set_weights(
                            actor_dict[next_automaton_state].get_weights())
                        target_critic_dict[next_automaton_state].set_weights(
                            critic_dict[next_automaton_state].get_weights())

                    reward = reward_flag - (np.cos(
                        (np.pi / 2) * reward_flag) * 0.3 + 0.02 * random.random())  # to break symmetry
                    buffer_dict[active_model].record((prev_state, action, reward, state))
                    episodic_reward += reward

                    buffer_dict[active_model].learn()
                    update_target(target_actor_dict[active_model].variables, actor_dict[active_model].variables, tau)
                    update_target(target_critic_dict[active_model].variables, critic_dict[active_model].variables, tau)

                    if next_automaton_state == -1:
                        break
                    if reward_flag > 0.5:
                        # print('reward reached!')
                        self.DFA.reset()

                    prev_state = state

                ep_reward_list.append(episodic_reward)

                avg_reward = np.mean(ep_reward_list[-40:])
                print("episode: {}, average episode reward={}".format(episode, avg_reward))
                self.q_at_initial_state.append(avg_reward)

            self.Q = actor_dict

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
                self.Q = actor_dict
                self.early_interruption = 1

    def reward(self, reward_flag):
        if reward_flag > 0:
            return 1
        else:
            return 0

    def action_space_augmentation(self):
        if self.DFA.automaton_state in self.DFA.epsilon_transitions.keys():
            product_MDP_action_space = self.MDP.action_space + \
                                       self.DFA.epsilon_transitions[self.DFA.automaton_state]
        else:
            product_MDP_action_space = self.MDP.action_space
        return product_MDP_action_space
