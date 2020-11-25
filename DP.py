from MDP import build_mazeMDP, print_policy
import numpy as np
import numpy as np


class DynamicProgramming:
    def __init__(self, MDP):
        self.R = MDP.R
        self.T = MDP.T
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    def helper(self, state, V):
        A = np.zeros(self.nActions)
        for action in range(self.nActions):  # take action
            for next_state in range(self.nStates):  # next state
                prob = self.T[action][state][next_state]
                reward = self.R[action][state]
                A[action] += prob * (reward + self.discount * V[next_state])
        return A

    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        iterId = 0
        V = np.zeros(self.nStates)
        while iterId < nIterations:
            delta = 0  # ref Sutton page 205
            # Compute expected value of each action
            for state in range(self.nStates):  # current state
                # compute best action value given state
                A = self.helper(state, V)
                v = np.max(A)  # max_a
                # max(delta, |v − V (s)|)
                delta = max(delta, np.abs(v - V[state]))
                V[state] = v  # update V
            if delta < tolerance:
                break
            iterId += 1

        # EVAL
        policy = self.extractPolicy(V)

        return [policy, V, iterId, delta]

    def policyIteration_v1(self, initialPolicy, nIterations=np.inf, tolerance=0.01):
        """Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar"""

        policy = np.zeros([self.nActions, self.nStates])
        policy[0] = 1
        iterId = 0
        # Loop
        while True:
            # Run policy eval
            V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
            policy_stable = True
            # loop for each s in S
            for state in range(self.nStates):
                # Find old action
                action = np.argmax(policy[:, state])
                # New action
                action_values = self.helper(state, V)
                best_action = np.argmax(action_values)
                # If old-action 6= ⇡(s), then policy-stable false
                if action != best_action:
                    policy_stable = False
                # update policy
                policy[:, state] = np.eye(self.nActions)[best_action]

            if policy_stable:  # return policy
                break
            iterId += 1  # increment

        policy = self.extractPolicy(V)

        return [policy, V, iterId]

    def extractPolicy(self, V):
        """Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries"""

        policy = np.zeros(self.nStates)
        for i in range(self.nStates):
            A = self.helper(i, V)
            best_action = np.argmax(A)
            policy[i] = best_action

        return policy

    def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
        """Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries"""

        V = np.zeros(self.nStates)
        # loop
        for state in range(self.nStates):
            # value to 0
            v = 0
            action = np.argmax(policy[:, state])
            for next_state, prob in enumerate(self.T[action][state]):
                v += prob * (self.R[action][state] + self.discount * V[next_state])
            V[state] = v
        return V

    def policyIteration_v2(
        self, nPolicyEvalIterations=10, nIterations=np.inf, tolerance=0.01
    ):
        # init policy, taking a_i = 0 in all states
        policy = np.zeros([self.nActions, self.nStates])
        policy[0] = 1
        iterId = 0
        # Loop
        while True:
            # Run policy eval
            V, delta = self.evaluatePolicy_IterativeUpdate(
                policy, tolerance, nPolicyEvalIterations
            )
            policy_stable = True
            # loop for each s in S
            for state in range(mdp.nStates):
                # Find old action
                action = np.argmax(policy[:, state])
                # New action
                action_values = self.helper(state, V)
                best_action = np.argmax(action_values)
                # If old-action 6= ⇡(s), then policy-stable false
                if action != best_action:
                    policy_stable = False
                # update policy
                policy[:, state] = np.eye(self.nActions)[best_action]

            if policy_stable:  # return policy
                break
            iterId += 1  # increment

        policy = np.array([np.argmax(policy[:, i]) for i in range(self.nStates)])
        return [policy, V, iterId, delta]

    def evaluatePolicy_IterativeUpdate(self, policy, epsilon, nIterations=10):
        V = np.zeros(self.nStates)
        # l
        for i in range(10):
            # delta to 0
            delta = 0
            # Loop over s in S
            for state in range(self.nStates):
                # value to 0
                v = 0
                for action, action_prob in enumerate(policy[:, state]):
                    for next_state, prob in enumerate(self.T[action][state]):
                        v += (
                            action_prob
                            * prob
                            * (self.R[action][state] + self.discount * V[next_state])
                        )
                delta = max(delta, np.abs(v - V[state]))
                V[state] = v
            if delta < epsilon:
                break
        return V, delta


if __name__ == "__main__":
    mdp = build_mazeMDP()
    dp = DynamicProgramming(mdp)
    # Test value iteration
    [policy, V, nIterations, epsilon] = dp.valueIteration(
        initialV=np.zeros(dp.nStates), tolerance=0.01
    )
    print_policy(policy)
    # # Test policy iteration v1
    # [policy, V, nIterations] = dp.policyIteration_v1(
    #     np.zeros(dp.nStates, dtype=int))
    # print_policy(policy)
    # Test policy iteration v2
    [policy, V, nIterations, epsilon] = dp.policyIteration_v2(
        np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), tolerance=0.01
    )
    print_policy(policy)
