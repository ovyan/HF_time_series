from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from learner_wrapper import DynamicsControlDeltaWrapper, DynamicsControlWrapper
from dad_control import DaDControl
from cartpole import CartPole
from gen_data import get_rand_train, get_rand_test
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# DAD_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
# sys.path.append(DAD_MODULE_PATH)

# print(DAD_MODULE_PATH)


# # Choose the system to use. Needs to follow a common api.
SYSTEM = CartPole


def x0_low_high(system):
    """Lower and higher bound for random intitial state generation. """
    if system == CartPole:
        x0_low = np.array([-1, -1, -np.pi, -0.5])
        x0_high = np.array([1, 1, np.pi, 0.5])
    else:
        raise Exception('No x0 bounds defined for this sytem')
    return x0_low, x0_high


def run_episodes(policy, num_episodes, T):
    num_episodes = 3
    T = 5
    """Generate num_episodes runs of the SYSTEM. """
    # Lower and higher bound for random intitial state generation.
    x0_low, x0_high = x0_low_high(SYSTEM)
    all_states = []
    all_actions = []
    for i in range(num_episodes):
        x0 = x0_low + (x0_high-x0_low)*np.random.random(4)
        states, actions = run_trial(policy, T, x0)
        all_states.append(states)
        all_actions.append(actions)
    print(all_actions)
    print(np.array(all_actions).shape)
    print(np.dstack(all_actions).transpose((1, 0, 2)))
    exit(0)
    return np.stack(all_states, axis=2), np.dstack(all_actions).transpose((1, 0, 2))


def run_trial(policy, T, x0=np.array((0, 0, np.pi/2., 0))):
    """Generate T timesteps of data from the SYSTEM. """
    # Initialize the system at x0.
    system = SYSTEM(x0)
    DT = 0.10  # simulate at 10 Hz
    xt = x0.copy()
    for t in range(T):
        ut = policy.u(xt)
        xt = system.step(DT, ut, sigma=1e-3+np.zeros(SYSTEM.state_dim()))
    X = system.get_states()
    U = system.get_controls()
    return X, U


class RandomLinearPolicy(object):
    """Generates a N-dimensional random control within a range. """

    def __init__(self, state_dim, control_dim, u_min=-1.0, u_max=1.0):
        self.A = np.random.random((control_dim, state_dim))

    def u(self, state):
        # return self.u_min + (self.u_max-self.u_min)*np.random.rand(self.control_dim)
        return np.dot(self.A, state)


def optimize_learner_dad(learner, X, U, iters, train_size=0.5):
    print(X.shape)
    print(U.shape)
    num_traj = X.shape[2]
    if train_size < 1.0:
        # from sklearn import cross_validation
        from sklearn.model_selection import train_test_split
        border = int(X.shape[2] * train_size)
        Xtrain = X[:, :, :border]
        Xtest = X[:, :, border:]
        Utrain = U[:, :, :border]
        Utest = U[:, :, border:]
    elif train_size == 1.0:
        Xtrain = X
        Xtest = X
        Utrain = U
        Utest = U
    else:
        raise Exception('Train size must be in (0,1]')

    dad = DaDControl()
    dad.learn(Xtrain, Utrain, learner, iters, Xtest, Utest, verbose=True)
    print(' DaD (iters:{:d}). Initial Err: {:.4g}, Best: {:.4g}'.format(iters,
                                                                        dad.initial_test_err, dad.min_test_error))
    return dad


if __name__ == "__main__":
    print('Defining the learner')
    # learner = DynamicsControlDeltaWrapper(MLPRegressor(hidden_layer_sizes=(20, 10),
    #                                                    activation='tanh', alpha=1e-3, max_iter=int(1e4), warm_start=False))
    # learner = DynamicsControlDeltaWrapper(Ridge(alpha=1e-4, fit_intercept=True))
    # learner = DynamicsControlWrapper(Ridge(alpha=1e-4, fit_intercept=True))
    # learner = DynamicsControlWrapper(LinearRegression(fit_intercept=True))
    learner = DynamicsControlWrapper(MLPRegressor(hidden_layer_sizes=(10, 4),
                                                  activation='tanh', alpha=1e-3, max_iter=int(1e4), warm_start=False))

    NUM_EPISODES = 50
    T = 50
    print('Generating train data')
    policy = RandomLinearPolicy(SYSTEM.state_dim(), SYSTEM.control_dim())

    Xtrain, Utrain = get_rand_train()
    # Xtrain, Utrain = run_episodes(policy, NUM_EPISODES, T)

    print('Generating test data')
    # Xtest, Utest = run_episodes(policy, NUM_EPISODES, T)
    Xtest, Utest = get_rand_test()

    print('\nLearning dynamics')
    iters = 25
    dad = optimize_learner_dad(learner, Xtrain, Utrain, iters, train_size=1)

    dad_pred, dad_err = dad.test(Xtest, Utest, dad.min_test_error_model)
    non_dad_pred, non_dad_err = dad.test(Xtest, Utest, dad.initial_model)

    plt.figure(figsize=[12, 7])
    plt.plot(np.vstack(dad_pred)[:, 0], label='dad_pred', marker='o')
    plt.plot(np.vstack(non_dad_pred)[:, 0], label='non_dad_pred')
    plt.plot(np.vstack(Xtest)[:, 0], label='test')
    # plt.plot(np.vstack(Xtrain)[:, 0], label='train')
    plt.legend(loc="upper right")
    # plt.ylim([-5, 5])
    plt.show()

    print(' Err without DaD: {:.4g}, Err with DaD: {:.4g}'.format(
        np.mean(non_dad_err), np.mean(dad_err)))

    # print(learner.info())
