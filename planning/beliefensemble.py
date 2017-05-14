import numpy as np
from polytope import *

class BeliefEnsemble:
    """ Belief ensemble
        
        As opposed to having a general simulator for the world,
        we will optimize our simulator to operate over ensembles
        of belief states for maximum speed.

        A single belief ensemble corresponds to one belief domain,
        which includes a single shape and the discretization
        of state space over <x,y,theta>.

        We will use the power of numpy matrices to compute the
        observation model for a given list F of finger-point
        locations simultaneously across an entire ensemble of
        object poses. Do this n ways for the n faces of the
        polygons, then AND these vectors together to determine
        whether interior/exterior, which determines Touch/NoTouch.
        """
    
    def __init__(self, poly, x_lim, y_lim, theta_lim, nx, ny, ntheta):
        """ Initializes a 1-shape discretization problem for our
            state-space. Can use this as an obs model for the various
            layers of our system (and for the sub-shapes making up
            a nonconvex shape), then fuse them together.

            Parameters
            ---
            shape: convex shape for which this domain is defined.
            
            x_lim: the limits of x-space, from 0-index end to
            (nx-1)-index end. A tuple (x0, x1), in continuous units.

            y_lim: the limits of y-space, from 0-index end to
            (ny-1)-index end. A tuple (y0, y1), in continuous units.

            theta_lim: the limits of theta-space, from 0-index
            end to (ntheta-1)-index end. A tuple (theta0, theta1),
            in continuous units.

            nx: the number of bins in x-space. 0-index will be at x0,
            (nx-1)-index will be at x1. Must be a positive int.

            ny: the number of bins in y-space. 0-index will be at y0,
            (ny-1)-index will be at y1. Must be a positive int.

            ntheta: the number of bins in theta-space. 0-index will
            be at theta0, (nx-1)-index will be at theta1. Must be a
            positive int.

            Returns
            ---
            New BeliefEnsemble simulator with the given
            specifications."""
        self.poly = poly
        self.x0, self.x1 = x_lim
        self.y0, self.y1 = y_lim
        self.theta0, self.theta1 = theta_lim
        self.nx = nx
        self.ny = ny
        self.ntheta = ntheta

        # Build necessary data structures from inputs
        self.S = self.nx * self.ny * self.ntheta

        # For every positioning of the object, make a polytope
        self.polytopes = []
        for i in range(self.S):
            itheta = i % (self.nx * self.ny)
            iy = (i // self.ntheta) % (self.nx)
            ix = (i // self.ntheta // self.ny)

            x = self.x0 + ix*((self.x1 - self.x0)/self.nx)
            y = self.y0 + iy*((self.y1 - self.y0)/self.ny)
            theta = self.theta0 + itheta*((self.theta1 - self.theta0)/self.ntheta)
            vt = np.array([[x],[y]])
            
            self.polytopes.append(self.poly.rotated_about_origin(theta).translated_by_2D_vector(vt))
    
    def get_uniform_belief(self):
        """ get a uniform belief over the states at the
            current belief discretization

            TODO Params and Return"""
        return np.ones(self.S, 1) / float(self.S)

    def get_ideal_obs(self, fingers):
        """ Gives a vector for the predicted O over each S

            Parameters
            ---
            fingers: list of k 2D Numpy vectors giving the
            absolute [[x],[y]] positions of the fingers            

            Returns
            ---
            S-by-k matrix with ones in the ith row and jth
            column if the ith state would yield contact in
            the jth finger.
            ."""
        S = self.S
        k = len(fingers)

        finger_vectors = \
                       [np.array([fingers[i][0],fingers[i][1],[1.0]]) \
                        for i in range(len(fingers))]

        obs = np.zeros((S, k))

        for i in range(S):
            for j in range(k):
                if self.polytopes[i].test_point_interior(finger_vectors[j]):
                    obs[i,j] = 1.0

        return obs

    def prob_obs(self, fingers, obs, accuracy=1.0):
        """ Vector of probability values in {p,(1-p)} giving
            whether the TODO

            TODO Params, Return. accuract in [0,1]"""
        ideal_obs = self.get_ideal_obs(fingers)

        prob_obs = np.zeros((self.S,1))

        for i in range(self.S):
            if np.sum(ideal_obs[i] == obs) == len(obs):
                prob_obs[i,0] = accuracy
            else:
                prob_obs[i,0] = 1-accuracy

        return prob_obs


if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    poly = Square(5.0, 5.0)
    x_lim = (0.0, 20.0)
    y_lim = (0.0, 20.0)
    theta_lim = (0.0, 0.0)
    nx = 15
    ny = 15
    ntheta = 1
    be_square_norot = BeliefEnsemble(poly, x_lim, y_lim, theta_lim, nx, ny, ntheta)


    # Use fingers in a row:
    fingers = [np.array([[7.0 + 2*float(x)],[13.0]]) for x in range(-1,1+1)]
    obs = be_square_norot.get_ideal_obs(fingers)
    print obs
    prob_obs = be_square_norot.prob_obs(fingers, [1, 1, 0])
    print np.transpose(np.reshape(prob_obs, (nx, ny)))

    print prob_obs[2*15+7]

    be_square_norot.polytopes[2*15+7].render_polytope(plt, xmax=20)
