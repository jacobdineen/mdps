import numpy as np


def e_soft(self, test, Q):
    def policy(state):
        a = np.ones(self.mdp.nActions) * epsilon / self.mdp.nActions
        best_action = np.argmax(Q[state, :])
        a[best_action] = 1 - ((self.mdp.nActions - 1) * epsilon / self.mdp.nActions)
        return a

    return policy


e_soft(test)
