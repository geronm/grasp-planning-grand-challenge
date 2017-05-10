import numpy as np
import random

class POMDP(object):
    """
    Partially Observable Markov Decision Process.

    This is an abstract class which should be implemented by
    subclasses corresponding to specific POMDP problems. In
    defining a POMDP problem instance, one must define a
    representation for each of the following datatypes:

    belief
        a belief over states for the POMDP

    action
        an action taken at any step in the POMDP

    observation
        an observation made at any step in the POMDP
    """
    def __init__(self):
        pass

    def prob_obs_given_bs_a(self, b_s, a, o):
        """
        Compute the probability of an observation given an action and belief
        state.

        Parameters
        ---
        b_s : belief

        a : action

        o : observation

        Returns
        ---
        prob : float

        """
        raise NotImplementedError('Not implemented.')

    def update_belief(self, b_s, a, o):
        """
        Update the belief state given an action and observation.

        Parameters
        ---
        b_s : belief

        a : action

        o : observation

        Returns
        ---
        b_s_new : belief

        """
        raise NotImplementedError('Not implemented.')

    def cost(self, b_s, actions, observations):
        """
        Compute the cost of the belief state given the actions taken and
        observations made so far. If the belief state is nonterminal, cost also
        includes the heuristic cost of the belief state.

        Parameters
        ---
        b_s : belief

        actions : list(action)

        observations : list(observation)

        Returns
        ---
        cost : float

        """
        raise NotImplementedError('Not implemented.')

    def is_terminal_belief(self, b_s, a, o):
        """
        Determine whether the belief state is terminal.

        Parameters
        ---
        b_s : belief

        a : action

        o : observation

        Returns
        ---
        is_terminal_belief : bool

        """
        raise NotImplementedError('Not implemented.')

    def heuristic(self, b_s):
        """
        Compute the heuristic cost of a given belief state. In general, the more
        uncertainty in the belief, the higher the cost. Cost should be positive.

        Parameters
        ---
        b_s : belief

        Returns
        ---
        heuristic_cost : float

        """
        raise NotImplementedError('Not implemented.')

    def get_possible_actions(self, b_s):
        """
        Get the possible actions given a belief state.

        Parameters
        ---
        b_s : belief

        Returns
        ---
        actions : list(action)

        """
        raise NotImplementedError('Not implemented.')

    def get_possible_observations(self, b_s, a):
        """
        Get the possible observations given a belief state and action.

        Parameters
        ---
        b_s : belief

        a : action

        Returns
        ---
        observations : list(observation)

        """
        raise NotImplementedError('Not implemented.')

    def solve(self, b_s_init, depth=3, max_actions=None):
        """
        Solve the POMDP instance using expectimax search.

        Parameters
        ---
        b_s_init : belief
            initial belief state

        depth : int
            max depth of the tree

        max_actions : int
            maximum number of actions to consider at each level of the
            tree, or None if all actions and observations should be
            considered.

        Returns
        ---
        out : float

        a0_best : action

        """
        root = ObservationNode(self, b_s_init, None, None)

        out, a0_best = self.alphabeta_observation_node(root, depth, float('-Inf'), float('+Inf'), max_actions)

        return out, a0_best

    def alphabeta_action_node(self, act_node, depth, alpha, beta, max_actions=None):
        """
        Performs alpha-beta search from an action node.
        """
        v = 0.0
        assert len(act_node.get_children()) > 0, 'No observations possible for action %s' % str(act_node.a)
        observations_list = act_node.get_children()
        if max_actions is not None:
            random.shuffle(observations_list)
            observations_list = observations_list[:max_actions]
        for child_obs in observations_list:
            prob_obs = self.prob_obs_given_bs_a(act_node.b_s, act_node.a, child_obs.o)
            if prob_obs > 0:
                value, a_best = self.alphabeta_observation_node(child_obs, depth, alpha, beta, max_actions)
                v += prob_obs*value
                if v < alpha:
                    break # alpha cut-off
        return v

    def alphabeta_observation_node(self, obs_node, depth, alpha, beta, max_actions=None):
        """
        Performs alpha-beta search from an observation node.
        """
        parent_action = None if (obs_node.parent_a is None) else obs_node.parent_a.a
        current_obs = obs_node.o
        assert len(obs_node.get_children()) > 0, 'No actions possible for belief'
        if depth == 0 or self.is_terminal_belief(obs_node.b_s, parent_action, current_obs):
            # Return negative cost so higher value is better
            return -self.cost(obs_node.b_s, obs_node.get_actions(), obs_node.get_observations()), None
        # If not terminal, iterate over next action nodes
        v = float('-Inf');
        a_best = None
        action_list = obs_node.get_children()
        if max_actions is not None:
            random.shuffle(action_list)
            action_list = action_list[:max_actions]
        for child_act in action_list:
            value = self.alphabeta_action_node(child_act, depth-1, alpha, beta, max_actions)
            if v < value:
                a_best = child_act.a
            v = max(v, value)
            alpha = max(v, alpha)
        return v, a_best


class ActionNode(object):
    """
    Represents an "action node", a search tree node for which an
    action has just been taken. These nodes are the intermediate
    nodes of the tree, whereas observation nodes correspond to
    fully-transitioned beliefs.
    """
    def __init__(self, pomdp, b_s, a, parent_o):
        """
        Constructs an instance of ActionNode

        Parameters
        ---
        pomdp : POMDP
            the pomdp that we're trying to search over.

        b_s : belief
            the belief just before action a was taken.

        a : action
            the action that was just taken.

        parent_o : ObservationNode
            the parent ObservationNode instance for this
            ActionNode. Must not be None, as ActionNodes are
            intermediate nodes.
        """
        self.pomdp = pomdp
        self.b_s = b_s
        self.a = a
        self.parent_o = parent_o

    def get_children(self):
        """
        Get ObservationNodes corresponding to the possible observations
        given this node's belief state and action.

        Returns
        ---
        child_nodes : list(ObservationNode)
            ObservationNode instances, with beliefs that have
            been transitioned based on the action a and
            the observation o.
        """
        child_nodes = [ObservationNode(self.pomdp, self.pomdp.update_belief(self.b_s, self.a, o), o, self) for o in self.pomdp.get_possible_observations(self.b_s, self.a)]
        return child_nodes


class ObservationNode(object):
    """
    Represents an "observation node", a search tree node for which an
    action and an observation have just been taken (or the root node
    of the tree). These nodes correspond to fully-transitioned beliefs.
    """
    def __init__(self, pomdp, b_s, o, parent_a):
        """
        Constructs an instance of ObservationNode

        Parameters
        ---
        pomdp : POMDP
            the pomdp that we're trying to search over.

        b_s : belief
            the belief just after action parent_a.a and observation
            o were processed.

        o : observation
            the observation that was just made.

        parent_a : ActionNode
            the parent ObservationNode instance for this
            ActionNode. May be None, in which case this
            node would represent the root of the search tree.
        """
        self.pomdp = pomdp
        self.b_s = b_s
        self.o = o
        self.parent_a = parent_a

    def get_children(self):
        """
        Get ActionNodes corresponding to the possible actions
        given this node's belief state.

        Returns
        ---
        child_nodes : list(ActionNode)
            ActionNode instances, with beliefs equal to ours
            and actions corresponding to the possible actions
            under our current belief.
        """
        if self.parent_a is not None:
            child_nodes = [ActionNode(self.pomdp,
                               self.b_s,
                               a,
                               self) for a in self.pomdp.get_possible_actions(self.b_s)]
        else:
            child_nodes = [ActionNode(self.pomdp,
                               self.b_s,
                               a,
                               self) for a in self.pomdp.get_possible_actions(self.b_s)]
        return child_nodes

    def get_observations(self):
        """
        Get all the observations made up to this point, in a
        list.

        Returns
        ---
        observations : list(observation)
            List of observations made up to this point in the
            search tree.
        """
        if self.parent_a.parent_o is None or self.parent_a.parent_o.o is None:
            return [self.o]
        return [self.o]          + self.parent_a.parent_o.get_observations()

    def get_actions(self):
        """
        Get all the actions taken up to this point, in a
        list.

        Returns
        ---
        actions : list(action)
            List of actions taken up to this point in the
            search tree.
        """
        if self.parent_a.parent_o is None or self.parent_a.parent_o.o is None:
            return [self.parent_a.a]
        return [self.parent_a.a] + self.parent_a.parent_o.get_actions()
