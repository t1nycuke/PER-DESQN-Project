import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pyESN_online import ESN
import copy

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class PRDDeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10,
            memory_size=300,
            lr=0.015,
            prioritized=True,
            sess=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = memory_size
        self.epsilon = e_greedy


        self.learn_step_counter = 0


        self.lr = lr

        self.prioritized = prioritized

        self.memory = Memory(capacity=memory_size)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        self._build_net()

        self.cost_his = []


    def _build_net(self):
        # ------------------ ESN parameters ------------------
        nInternalUnits = 64
        spectralRadius = 0.80
        inputScaling = 2 * np.ones(self.n_features)
        inputShift = -1 * np.ones(self.n_features)
        teacherScaling = 1 * np.ones(self.n_actions)
        teacherShift = 0 * np.ones(self.n_actions)
        self.nForgetPoints = 50


        self.eval_net = ESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                            spectral_radius=spectralRadius, sparsity=1 - min(0.2 * nInternalUnits, 1), noise=0,
                            lr=self.lr,
                            input_shift=inputShift, input_scaling=inputScaling,
                            teacher_scaling=teacherScaling, teacher_shift=teacherShift)
        self.ISWeights = tf.placeholder(tf.float32, [None, 1])  # 输入
        # ------------------ build target_net ------------------
        self.target_net = ESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                              spectral_radius=spectralRadius, sparsity=1 - min(0.2 * nInternalUnits, 1), noise=0,
                              lr=self.lr,
                              input_shift=inputShift, input_scaling=inputScaling,
                              teacher_scaling=teacherScaling, teacher_shift=teacherShift)

        self.target_net = copy.deepcopy(self.eval_net)
    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)

    def choose_action(self, observation):

        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:

            actions_value = self.eval_net.predict(observation, 0, continuation=True)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):


        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        eval_net_input = batch_memory[:, :self.n_features]
        target_net_input = batch_memory[:, -self.n_features:]

        q_eval = self.eval_net.predict(eval_net_input, 0, continuation=False)
        q_next = self.target_net.predict(target_net_input, 0, continuation=False)

        actions_value = self.eval_net.predict(target_net_input, 0, continuation=True)
        action = np.argmax(actions_value, axis = 1)


        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]


        next_q_value = self.gamma * q_next[batch_index, action]



        for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]


        pred_train = self.eval_net.fit(eval_net_input, q_target, self.nForgetPoints)
        self.cost = ISWeights * np.linalg.norm(pred_train - q_target)
        self.abs_errors = tf.reduce_sum(tf.abs(q_target - pred_train), axis=1)

        abserrors = self.sess.run([self.abs_errors])

        for i in range(len(abserrors)):
         self.memory.batch_update(tree_idx, abserrors[i])
        self.cost_his.append(self.cost)


        self.eval_net.startstate = copy.deepcopy(self.eval_net.laststate)
        self.eval_net.startinput = copy.deepcopy(self.eval_net.lastinput)
        self.eval_net.startoutput = copy.deepcopy(self.eval_net.lastoutput)

        self.target_net.startstate = copy.deepcopy(self.target_net.laststate)
        self.target_net.startinput = copy.deepcopy(self.target_net.lastinput)
        self.target_net.startoutput = copy.deepcopy(self.target_net.lastoutput)


        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net = copy.deepcopy(self.eval_net)


        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
