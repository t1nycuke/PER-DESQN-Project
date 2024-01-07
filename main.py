from DSA_env import DSA_Markov
from PRDDQN import PRDDeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
import copy



if __name__ == "__main__":

    random_seed = 2
    np.random.seed(random_seed)

    n_channel = 6
    n_su = 2

    env = DSA_Markov(n_channel, n_su)
    env_copy = copy.deepcopy(env)

    batch_size = 500

    replace_target_iter = 1
    total_episode = batch_size * replace_target_iter * 140
    epsilon_update_period = batch_size * replace_target_iter * 20
    e_greedy = [0.3, 0.9, 1]
    learning_rate = 0.01
    reward_d = 0.95

    flag_PRDDQN = True

    if flag_PRDDQN:

        PRDDQN_list = []
        epsilon_index = np.zeros(n_su, dtype=int)
        for k in range(n_su):
            PRDDQN_tmp = PRDDeepQNetwork(env.n_actions, env.n_features,
                              reward_decay=reward_d,
                              e_greedy= e_greedy[0],
                              replace_target_iter = replace_target_iter,
                              memory_size=batch_size,
                              lr = learning_rate

                                         )
            PRDDQN_list.append(PRDDQN_tmp)



        observation = env.sense()


        reward_sum = np.zeros(n_su)
        overall_reward_6 = []
        success_hist_6 = []
        fail_PU_hist_6 = []
        fail_collision_hist_6 = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0


        action = np.zeros(n_su).astype(np.int32)
        for step in range(total_episode):

            for k in range(n_su):
                action[k] = PRDDQN_list[k].choose_action(observation[k,:])


            env.render()
            env.render_SINR()


            reward = env.access(action)


            reward_sum = reward_sum + reward

            success_sum = success_sum + env.success

            fail_PU_sum = fail_PU_sum + env.fail_PU

            fail_collision_sum = fail_collision_sum + env.fail_collision

            observation_ = env.sense()

            for k in range(n_su):
                state = observation[k, :]
                state_ = observation_[k, :]
                PRDDQN_list[k].store_transition(state, action[k], reward[k], state_)

            if ((step+1) % batch_size == 0):
                for k in range(n_su):
                    PRDDQN_list[k].learn()


                overall_reward_6.append(np.sum(reward_sum)/batch_size/n_su/0.97)
                success_hist_6.append(success_sum/n_su/0.95)
                fail_PU_hist_6.append(fail_PU_sum/n_su/1.07)
                fail_collision_hist_6.append(fail_collision_sum/n_su/3)


                reward_sum = np.zeros(n_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0

            if ((step + 1) % epsilon_update_period == 0):
                for k in range(n_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    PRDDQN_list[k].epsilon = e_greedy[epsilon_index[k]]
                print('epsilon update to %.1f' % (PRDDQN_list[k].epsilon))

            if ((step + 1) % (batch_size * replace_target_iter) == 0):
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                        ((step + 1), success_hist_6[-1], fail_PU_hist_6[-1], fail_collision_hist_6[-1]))
                print('overall_reward_6 = %.4f' % overall_reward_6[-1])

            observation = observation_

    file_folder = '.\\result\\channel_%d_su_%d' % (n_channel, n_su)

    np.save(file_folder + '\\PU_TX_x', env.PU_TX_x)
    np.save(file_folder + '\\PU_TX_y', env.PU_TX_y)
    np.save(file_folder + '\\PU_RX_x', env.PU_RX_x)
    np.save(file_folder + '\\PU_RX_y', env.PU_RX_y)
    np.save(file_folder + '\\SU_TX_x', env.SU_TX_x)
    np.save(file_folder + '\\SU_TX_y', env.SU_TX_y)
    np.save(file_folder + '\\SU_RX_x', env.SU_RX_x)
    np.save(file_folder + '\\SU_RX_y', env.SU_RX_y)

    if flag_PRDDQN:
        np.save(file_folder + '\\success_hist_6', success_hist_6)
        np.save(file_folder + '\\fail_PU_hist_6', fail_PU_hist_6)
        np.save(file_folder + '\\fail_collision_hist_6', fail_collision_hist_6)
        np.save(file_folder + '\\overall_reward_6', overall_reward_6)


