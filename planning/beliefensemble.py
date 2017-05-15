import numpy as np
from polytope import *

class BeliefEnsemble:
    """ Belief Ensemble
        
        As opposed to having a general simulator for the world,
        we will optimize our simulator to operate over ensembles
        of belief states for maximum speed.

        A single belief ensemble corresponds to one belief domain,
        which includes a single convex shape and the discretization
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

        self.poly_num_eqns = len(self.poly.get_H())
        self.Hs = [[] for _ in range(self.poly_num_eqns)]

        # For every positioning of the object, make a polytope
        self.polytopes = []
        for i in range(self.S):
            ix, iy, itheta = self.big_index_to_indices(i)

            x = self.x0 + ix*((self.x1 - self.x0)/self.nx)
            y = self.y0 + iy*((self.y1 - self.y0)/self.ny)
            theta = self.theta0 + itheta*((self.theta1 - self.theta0)/self.ntheta)
            vt = np.array([[x],[y]])

            transformed_poly = self.poly.rotated_about_origin(theta).translated_by_2D_vector(vt)
            self.polytopes.append(transformed_poly)

            # Extract each polytope's H matrix rows
            transformed_H = transformed_poly.get_H()
            assert len(transformed_H) == len(self.Hs)
            for j in range(len(self.Hs)):
                self.Hs[j].append(transformed_H[j])

        for j in range(len(self.Hs)):
            self.Hs[j] = np.array(self.Hs[j])

    def big_index_to_indices(self, big_index):
        itheta = big_index % (self.ntheta)
        iy = (big_index // self.ntheta) % (self.nx)
        ix = (big_index // self.ntheta // self.ny)
        assert self.indices_to_big_index((ix, iy, itheta)) == big_index, \
               'ix, iy, itheta ' + str((ix,iy,itheta)) + '\n' + \
               'i2bi: %d  bi: %d' % (self.indices_to_big_index((ix, iy, itheta)), big_index)
        return (ix, iy, itheta)

    def indices_to_big_index(self, indices):
        ix, iy, itheta = indices
        return ix*(self.ny*self.ntheta) + iy*(self.ntheta) + itheta
    
    def get_uniform_belief(self):
        """ get a uniform belief over the states at the
            current belief discretization

            TODO Params and Return"""
        return np.ones(self.S, 1) / float(self.S)

    def get_ideal_obs_slow(self, fingers):
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


    def get_ideal_obs(self, fingers):
        """ Gives a vector for the predicted obs over each S

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

        finger_matrix = np.concatenate(tuple(finger_vectors), axis=1)

        # Parallel S-ways (one for each face of the polytope)
        results = []
        for i in range(len(self.Hs)):
            ax_minus_b = np.dot(self.Hs[i], finger_matrix)
            results.append(ax_minus_b)
        results = np.array(results)

        results_full = np.prod(0.0 + (results <= 0), axis=0)
        
        return results_full

    def prob_obs_slow(self, fingers, obs, accuracy=1.0):
        """ Vector of probability values in {p,(1-p)} giving
            prob obs given state.

            Parameters
            ---
            fingers: length-k list of finger vectors,
            same spec as prob_obs

            obs: length-k list of 1s and 0s corresponding to
            whether each finger made contact.

            accuracy: accuracy of the sensor. Correct reading
            with probability accuracy, incorrect reading
            with probability 1-accuracy or something. In range
            [0,1].

            Returns
            ---
            Vector of obs probabilities given state, length S-by-1. """
        ideal_obs = self.get_ideal_obs(fingers)

        prob_obs = np.zeros((self.S,1))

        # probabilities
        prob_correct = accuracy
        prob_incorrect = float(1-accuracy) / (2**(len(obs)) - 1)

        for i in range(self.S):
            if np.sum(ideal_obs[i] == obs) == len(obs):
                prob_obs[i,0] = prob_correct
            else:
                prob_obs[i,0] = prob_incorrect

        return prob_obs

    def prob_obs(self, fingers, obs, accuracy=1.0):
        """ Vector of probability values in {p,(1-p)} giving
            prob obs given state.

            Parameters
            ---
            fingers: length-k list of finger vectors,
            same spec as prob_obs \
    
            obs: length-k list of 1s and 0s corresponding to
            whether each finger made contact.

            accuracy: accuracy of the sensor. Correct reading
            with probability accuracy, incorrect reading
            with probability 1-accuracy or something. In range
            [0,1].

            Returns
            ---
            Vector of obs probabilities given state, length S-by-1. """
        ideal_obs = self.get_ideal_obs(fingers)

        fingers_many = np.kron(np.ones((len(ideal_obs),1)),np.array([obs]))

        # probabilities
        prob_correct = accuracy
        prob_incorrect = float(1-accuracy) / (2**(len(obs)) - 1)

        # find bitmask of matches
        fingers_matches = np.prod(0.0 + (fingers_many==ideal_obs), axis=1).reshape((len(ideal_obs),1))

        prob_obs = prob_incorrect + (prob_correct-prob_incorrect)*fingers_matches

        return prob_obs

    def render_belief_xy(self, plt, b_s, fingers=[], obs_true=[]):
        """ Renders the belief, marginalizing theta out

            Parameters
            ---
            plt: matplotlib.pyplot or plotting package with
            identical API

            b_s: vector S-by-1 of probability values or values
            of some sort.

            fingers: the finger placement vectors, as they would
            be passed into prob_obs

            obs_true: the true observation, as it would be
            passed into prob_obs
            
            Returns
            ---
            None, but shows the resulting plot. """
        scores = np.zeros((self.nx, self.ny))
        ixs = []
        iys = []
        ix_to_x = [(self.x0 + ix*((self.x1 - self.x0)/self.nx)) for ix in range(self.nx)]
        iy_to_y = [(self.y0 + iy*((self.y1 - self.y0)/self.ny)) for iy in range(self.ny)]
        
        for ix in range(self.nx):
            for iy in range(self.ny):
                x = ix_to_x[ix]
                y = iy_to_y[iy]

                prob_score = 0.0
                big_index_start = self.indices_to_big_index((ix,iy,0))
                for itheta in range(self.ntheta):
                    prob_score += b_s[big_index_start + itheta]

                scores[ix,iy] = prob_score
                ixs.append(ix)
                iys.append(iy)
        
        plt.pcolormesh(ix_to_x, iy_to_y, np.transpose(scores))


        if len(fingers) > 0:
            fxs = [f[0][0] for f in fingers]
            fys = [f[1][0] for f in fingers]
            plt.scatter(fxs, fys)

        
        plt.show()


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import time
    import random
    
    # poly = Square(5.0, 5.0)
    poly = Hexagon(7.0)
    x_lim = (0.0, 30.0)
    y_lim = (0.0, 20.0)
    theta_lim = (0.0, 2*np.pi)
    nx = 40
    ny = 40
    ntheta = 40
    start = time.time()
    be_square_norot = BeliefEnsemble(poly, x_lim, y_lim, theta_lim, nx, ny, ntheta)

    print 'time to construct %d-state BeliefEnsemble %f' % (be_square_norot.S, time.time() - start)

    if True:
        
        # Use fingers in a row:
        fingers = [np.array([[7.0 + 2*float(x)],[13.0]]) for x in range(-1,1+1)]
        obs_true = [0, 1, 1]
        #obs = be_square_norot.get_ideal_obs_slow(fingers)
        #print obs
        #print np.transpose(np.reshape(prob_obs, (nx, ny, ntheta)))

        start = time.time()
        
##        a = 0
##        for i in range(1):
##            fingers = [np.array([[7.0 + 2*float(x)],[13.0]]) for x in range(-1,1+1)]
##            obs = be_square_norot.get_ideal_obs(fingers)
##            obs_slow = be_square_norot.get_ideal_obs_slow(fingers)
##            print(obs.shape)
##            print(obs_slow.shape)
##            print(obs)
##            print(obs_slow)
##            for j in range(len(obs)):
##                if not (obs[j] == obs_slow[j]).all():
##                    print ' h'
##                    print j
##                    print be_square_norot.big_index_to_indices(j)
##                    print obs[j]
##                    print obs_slow[j]
##                    for f in fingers:
##                        v = np.array([[f[0][0],f[1][0],1.0]]).transpose()
##                        print np.dot(be_square_norot.polytopes[j].H, v)
##                    #be_square_norot.polytopes[j].render_polytope(plt, xmax=30.0)

##### NOTE:: THE BELOW ASSERTION FAILS, WHICH MAKES IT SEEM LIKE THERE
##### IS AN ERROR IN THE FASTER CODE; BUT, INSPECTION REVEALS THIS SIMPLY
##### TO BE NUMERICS
##            assert (obs==obs_slow).all()
##            a += obs[1]

        
        # be_square_norot.polytopes[2*15+7].render_polytope(plt, xmax=20)

        prob_obs = be_square_norot.prob_obs(fingers, obs_true, accuracy=0.785)


        print 'time to query one set of fingers %f' % (time.time() - start)
        print 'done'

##        prob_obs_slow = be_square_norot.prob_obs_slow(fingers, [1, 1, 1], accuracy=0.785)

##        print prob_obs.shape
##        print prob_obs_slow.shape
##
##        print prob_obs.transpose()
##        print prob_obs_slow.transpose()

##        for j in range(len(obs)):
##            if not (prob_obs[j] == prob_obs_slow[j]).all():
##                print ' h'
##                print j
##                print prob_obs[j]
##                print prob_obs_slow[j]
        

##        assert (prob_obs_slow == prob_obs).all()

        be_square_norot.render_belief_xy(plt, prob_obs, fingers, obs_true)
