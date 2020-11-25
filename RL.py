from MDP import build_mazeMDP, print_policy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

rcParams["figure.figsize"] = 15, 5


class ReinforcementLearning:
    def __init__(self, mdp, sampleReward):
        """
        Constructor for the RL class

        :param mdp: Markov decision process (T, R, discount)
        :param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
        the mean of the distribution and returns a sample from the distribution.
        """

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        """Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        """

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def OffPolicyTD(self, nEpisodes, epsilon=0.5):

        stats = {}
        Q = np.zeros([self.mdp.nActions, self.mdp.nStates])
        policy = np.zeros(self.mdp.nStates, int)

        for i_episode in range(1, nEpisodes + 1):
            rewards, episode_length, state = 0, 0, 0
            done = False

            while not done:
                choice = np.random.choice([0, 1], p=[epsilon, 1 - epsilon])
                if choice == 0:
                    action = np.random.choice(self.mdp.nActions)
                else:
                    action = policy[state]

                reward, next_state = self.sampleRewardAndNextState(state, action)

                best_next = np.argmax(Q[:, next_state])
                td_target = reward + self.mdp.discount * Q[best_next][next_state]
                td_delta = td_target - Q[action][state]
                Q[action][state] += 0.5 * td_delta

                for i in range(self.mdp.nStates):
                    policy[i] = np.argmax(Q[:, i])

                if self.mdp.isTerminal(state):
                    stats[i_episode] = (rewards, episode_length)
                    break
                else:
                    state = next_state
                    rewards += reward
                    episode_length += 1

        return [Q, policy, stats]

    def e_soft(self, epsilon, Q):
        def policy(state):
            a = np.ones(self.mdp.nActions) * epsilon / self.mdp.nActions
            best_action = np.argmax(Q[state, :])
            a[best_action] = 1 - ((self.mdp.nActions - 1) * epsilon / self.mdp.nActions)
            return a

        return policy

    def OffPolicyMC(self, nEpisodes, epsilon=0.0):
        stats = {}
        Q = np.zeros([self.mdp.nStates, self.mdp.nActions])
        C = np.zeros([self.mdp.nStates, self.mdp.nActions])
        b = self.e_soft(epsilon, Q)

        for i_episode in range(1, nEpisodes + 1):
            if i_episode % 1000 == 0:
                print(f"Episode {i_episode}/{nEpisodes}")

            # Behavioral Policy
            # Generate an episode using b: S0,A0,R1, . . . ,ST−1,AT−1,RT
            episode = []
            rewards, episode_length, state = 0, 0, 0
            state = 0
            while True:
                probs = b(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                reward, next_state = self.sampleRewardAndNextState(state, action)

                episode.append((state, action, reward))

                if self.mdp.isTerminal(state):
                    break
                state = next_state
                rewards += reward
                episode_length += 1

            stats[i_episode] = (rewards, episode_length)

            G = 0
            W = 1
            # Loop for each step of episode, t = T −1, T −2, . . . , 0:
            for idx, step in enumerate(episode[::-1]):
                # extract SAR
                state, action, reward = step

                # G = gamma * G + Rt+1
                G = self.mdp.discount * G + reward

                # C(St,At) <- C(St,At) +W
                C[state][action] += W

                # Q(St,At) <-  Q(St,At) + W/C(St,At) [G − Q(St,At)]
                Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

                if action != np.argmax(Q[state, :]):
                    break
                W = W * (1 / b(state)[action])

        policy = np.array([np.argmax(Q[i, :]) for i in range(self.mdp.nStates)])
        return [Q, policy, stats]


if __name__ == "__main__":
    mdp = build_mazeMDP()
    rl = ReinforcementLearning(mdp, np.random.normal)

    # # Test Q-learning
    # [Q, policy] = rl.OffPolicyTD(nEpisodes=500, epsilon=0.1)
    # print_policy(policy)

    # Test Off-Policy MC
    # [Q, policy,stats] = rl.OffPolicyMC(nEpisodes=25000, epsilon=0.1)
    # print_policy(policy)
    num_runs = 1
    num_episodes = 500

    dfs = pd.DataFrame()
    for run in range(num_runs):
        [Q, policy, stats] = rl.OffPolicyMC(nEpisodes=num_episodes, epsilon=0.1)
        df = pd.DataFrame.from_dict(stats).T
        dfs = dfs.append(df)

    dfs = dfs.reset_index()
    dfs = dfs.groupby("index").mean().reset_index()
    dfs.columns = ["episode", "mean_reward", "mean_steps"]

    sns.lineplot(x=dfs["episode"], y=dfs["mean_reward"])
    plt.xlabel("Episode")
    plt.title(f"Mean Reward over {num_runs} runs by episode")
    plt.ylim(-100, 100)
    plt.show()

    sns.lineplot(x=dfs["episode"], y=dfs["mean_steps"])
    plt.xlabel("Episode")
    plt.title(f"Mean episode length over {num_runs} runs by episode")
    plt.ylim(0, 250)

    plt.show()
