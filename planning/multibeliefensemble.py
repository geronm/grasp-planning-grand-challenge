import numpy as np
from polytope import *
from beliefensemble import BeliefEnsemble

class MultiBeliefEnsemble:
    """ Multi-Belief Ensemble
        
        As opposed to having a general simulator for the world,
        we will optimize our simulator to operate over ensembles
        of belief states for maximum speed.

        Multi-belief ensembles capture more complex scenarios
        than the simple belief ensemble. Whereas the former
        could handle only a single convex 2D shape in the 2D
        plane the multi-belief ensemble can handle unions
        of convex objects that share the same state-space in
        order to make nonconvex shapes

        (eg. the tetris angle:  #
                                ##   )

        This class leverages BeliefEnsemble's ability to query
        for the predicted finger-contacts accross all state
        space for a single convex shape. It can take an OR
        over these contacts to get the predicted contact
        observations for a union of convex shapes.
        """
    
    def __init__(self, belief_ensembles):
        """ Initializes a multi-shape discretization problem for our
            state-space. Can use this as an obs model for the various
            layers of our system (and for the sub-shapes making up
            a nonconvex shape). Fuses them together.

            Parameters
            ---
            belief_ensembles: a list of belief
            ensembles, whose convex shapes are
            OR'd together to give our shape. These
            ensembles must share all discretization
            configuration parameters. Must be nonempty
            

            Returns
            ---
            New BeliefEnsemble simulator with the given
            specifications."""

        be = belief_ensembles[0]
        x_lim, y_lim, theta_lim, nx, ny, ntheta = (be.x0, be.x1), \
                                                  (be.y0, be.y1), \
                                                  (be.theta0, be.theta1), \
                                                  be.nx, \
                                                  be.ny, \
                                                  be.ntheta
        for be in belief_ensembles[1:]:
            old = x_lim, y_lim, theta_lim, nx, ny, ntheta
            x_lim, y_lim, theta_lim, nx, ny, ntheta = (be.x0, be.x1), \
                                                  (be.y0, be.y1), \
                                                  (be.theta0, be.theta1), \
                                                  be.nx, \
                                                  be.ny, \
                                                  be.ntheta
            new = x_lim, y_lim, theta_lim, nx, ny, ntheta
            assert old == new, 'Different:\n%s\n%s' % (str(old),str(new))

        self.belief_ensembles = belief_ensembles

        # Build necessary data structures from inputs
        be = self.belief_ensembles[0]
        (self.x0, self.x1) = (be.x0, be.x1)
        (self.y0, self.y1) = (be.y0, be.y1)
        (self.theta0, self.theta1) = (be.theta0, be.theta1)
        self.nx = be.nx
        self.ny = be.ny
        self.ntheta = be.ntheta
        
        self.S = self.nx * self.ny * self.ntheta

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
        return np.ones((self.S, 1)) / float(self.S)

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

        results_full = self.belief_ensembles[0].get_ideal_obs(fingers)
        for i in range(len(self.belief_ensembles)):
            results_full = np.logical_or(results_full, self.belief_ensembles[i].get_ideal_obs(fingers))
        
        return 0.0 + results_full

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




class LayeredBeliefEnsemble:
    """ Layered Belief Ensemble
        
        As opposed to having a general simulator for the world,
        we will optimize our simulator to operate over ensembles
        of belief states for maximum speed.

        Layered belief ensembles capture more complex scenarios
        than the simple or multi belief ensemble. Whereas the former
        could handle only a single 2D shape in the 2D
        plane the multi-belief ensemble can reason about multiple
        2D planes above one another.

        The discrete planes will represent the TOP SURFACE of
        each layer. The 0-layer will be the highest (first-touched),
        while the (len()-1)-layer will be the lowest (last-touched,
        but noto table height). In a generative
        sense, the top layers will be hit first.
        
        This class leverages MultiBeliefEnsemble's ability to query
        for the predicted finger-contacts accross all state
        space for a single convex shape. It can take an OR
        over these contacts to get the predicted contact
        observations for a union of convex shapes. It can
        also maintain several such unions in parallel to
        model a 3D-layered observation domain.
        """
    
    def __init__(self, belief_ensembles, z_lim):
        """ Initializes a multi-shape discretization problem for our
            state-space. Can use this as an obs model for the various
            layers of our system (and for the sub-shapes making up
            a nonconvex shape). Fuses them together.

            Parameters
            ---
            belief_ensembles: a list of multibeliefensembles
            belief_ensemble[k] corresponds to the kth layer
            of our layered object. The length of this list
            will determine self.nz, the discretization size
            in the z dimension.

            z_lim: the vertical limits (lo,hi) between which our
            domain shall be discretized (lo-inclusive-hi-exclusive).

            Returns
            ---
            New BeliefEnsemble simulator with the given
            specifications."""

        
        be = belief_ensembles[0]
        x_lim, y_lim, theta_lim, nx, ny, ntheta = (be.x0, be.x1), \
                                                  (be.y0, be.y1), \
                                                  (be.theta0, be.theta1), \
                                                  be.nx, \
                                                  be.ny, \
                                                  be.ntheta
        for be in belief_ensembles[1:]:
            old = x_lim, y_lim, theta_lim, nx, ny, ntheta
            x_lim, y_lim, theta_lim, nx, ny, ntheta = (be.x0, be.x1), \
                                                  (be.y0, be.y1), \
                                                  (be.theta0, be.theta1), \
                                                  be.nx, \
                                                  be.ny, \
                                                  be.ntheta
            new = x_lim, y_lim, theta_lim, nx, ny, ntheta
            assert old == new, 'Different:\n%s\n%s' % (str(old),str(new))

        self.belief_ensembles = belief_ensembles


        # Build necessary data structures from inputs
        self.nz = len(self.belief_ensembles)
        (self.z0, self.z1) = z_lim

        be = self.belief_ensembles[0]
        (self.x0, self.x1) = (be.x0, be.x1)
        (self.y0, self.y1) = (be.y0, be.y1)
        (self.theta0, self.theta1) = (be.theta0, be.theta1)
        self.nx = be.nx
        self.ny = be.ny
        self.ntheta = be.ntheta
        
        self.S = self.nx * self.ny * self.ntheta

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

    def indices_to_continuous_pose(self, indices):
        ix, iy, itheta = indices
        
        x = self.x0 + (self.x1-self.x0)*ix
        y = self.y0 + (self.y1-self.y0)*iy
        theta = self.theta0 + (self.theta1-self.theta0)*itheta

        return (x, y, theta)

    def continuous_pose_to_indices(self, pose):
        x, y, theta = pose
        
        ix = (x - self.x0) // (self.x1-self.x0)
        iy = (y - self.y0) // (self.y1-self.y0)
        itheta = (theta - self.theta0) // (self.theta1-self.theta0)

        return (ix, iy, itheta)
    
    def z_index_to_continuous(self, iz):
        z = self.z0 + (self.z1-self.z0)*iz
        return float(z)

    def z_continuous_to_index(self, z):
        iz = (z - self.z0) // (self.z1-self.z0)
        return iz
    
    def get_uniform_belief(self):
        """ get a uniform belief over the states at the
            current belief discretization

            TODO Params and Return"""
        return np.ones((self.S, 1)) / float(self.S)

    def get_ideal_obs(self, fingers, iz):
        """ Gives a vector for the predicted obs over each S

            Parameters
            ---
            fingers: list of k 2D Numpy vectors giving the
            absolute [[x],[y]] positions of the fingers       

            iz: index corresponding to z-location of all
            the fingers (ie. they must
            be horizontally coplanar), an int

            Returns
            ---
            S-by-k matrix with ones in the ith row and jth
            column if the ith state would yield contact in
            the jth finger.
            ."""
        S = self.S
        k = len(fingers)

        # query in z layer (this is ideal!)
        results_full = self.belief_ensembles[iz].get_ideal_obs(fingers)
        
        return 0.0 + results_full

    def prob_obs(self, fingers, obs, b_z, accuracy=0.8):
        """ Vector of probability values in {p,(1-p)} giving
            prob obs given state.

            Parameters
            ---
            fingers: length-k list of finger vectors,
            same spec as prob_obs \
    
            obs: length-k list of 1s and 0s corresponding to
            whether each finger made contact.

            b_z: nz-by-1 Numpy vector giving the probability
            distribution belief over iz. Really ought to be
            eta-normalized, though won't complain if not.

            accuracy: accuracy of the sensor. Correct reading
            with probability accuracy, incorrect reading
            with probability 1-accuracy or something. In range
            [0,1].

            Returns
            ---
            Vector of obs probabilities given state, length S-by-1. """

        assert b_z.shape == (self.nz, 1)
        
        prob_obs = np.zeros((self.S, 1))

        for iz in range(self.nz):
            if b_z[iz][0] != 0:
                ideal_obs = self.get_ideal_obs(fingers, iz)

                fingers_many = np.kron(np.ones((len(ideal_obs),1)),np.transpose(np.array(obs)))

                # probabilities
                prob_correct = accuracy
                prob_incorrect = float(1-accuracy) / (2**(len(obs)) - 1)

                # find bitmask of matches
                fingers_matches = np.prod(0.0 + np.equal(fingers_many,ideal_obs), axis=1).reshape((len(ideal_obs),1))

                prob_obs += b_z[iz][0]*(prob_incorrect + (prob_correct-prob_incorrect)*fingers_matches)

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
    
    x_lim = (0.0, 50.0)
    y_lim = (0.0, 50.0)
    theta_lim = (0.5, 2*np.pi)
    nx = 160
    ny = 160
    ntheta = 1

    belief_ensembles = []

    poly1 = Square(5.0, 5.0)
    start = time.time()
    be_square_norot1 = BeliefEnsemble(poly1, x_lim, y_lim, theta_lim, nx, ny, ntheta)

    poly2 = Square(5.0, 5.0).translated_by_2D_vector(np.array([[10.0],[0.0]]))
    be_square_norot2 = BeliefEnsemble(poly2, x_lim, y_lim, theta_lim, nx, ny, ntheta)

    poly3 = Square(5.0, 5.0).translated_by_2D_vector(np.array([[0.0],[10.0]]))
    be_square_norot3 = BeliefEnsemble(poly3, x_lim, y_lim, theta_lim, nx, ny, ntheta)
    
    be_main = MultiBeliefEnsemble([be_square_norot1, be_square_norot2, be_square_norot3])

    belief_ensembles.append(be_main)
    
    poly1 = Square(5.0, 5.0)
    start = time.time()
    be_square_norot1 = BeliefEnsemble(poly1, x_lim, y_lim, theta_lim, nx, ny, ntheta)

    poly2 = Square(5.0, 5.0).translated_by_2D_vector(np.array([[-10.0],[0.0]]))
    be_square_norot2 = BeliefEnsemble(poly2, x_lim, y_lim, theta_lim, nx, ny, ntheta)

    poly3 = Square(5.0, 5.0).translated_by_2D_vector(np.array([[0.0],[-10.0]]))
    be_square_norot3 = BeliefEnsemble(poly3, x_lim, y_lim, theta_lim, nx, ny, ntheta)
    
    be_main = MultiBeliefEnsemble([be_square_norot1, be_square_norot2, be_square_norot3])

    belief_ensembles.append(be_main)

    be_main = LayeredBeliefEnsemble(belief_ensembles, (0.0, 1.0))

    print 'time to construct %d-state BeliefEnsemble %f' % (be_main.S, time.time() - start)

    if True:
        
        # Use fingers in a row:
        fingers = [np.array([[10.0 + 2*float(x)],[13.0]]) for x in range(-0,0+1)]
        obs_true = [1, 1]
        #obs = be_main.get_ideal_obs_slow(fingers)
        #print obs
        #print np.transpose(np.reshape(prob_obs, (nx, ny, ntheta)))

        start = time.time()

        prob_obs = be_main.prob_obs(fingers, obs_true, np.array([[0.5],[0.5]]), accuracy=0.785)


        print 'time to query one set of fingers %f' % (time.time() - start)
        print 'done'

        be_main.render_belief_xy(plt, prob_obs, fingers, obs_true)
