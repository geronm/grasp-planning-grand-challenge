import numpy as np
import matplotlib.pyplot as plt

# This class contains utility methods for constructing
# planar polygons (expressed in continuous coordinates)


class Polytope:
    """ Polytope

        Defines a polytope in the plane. A polytope
        is defined as the intersection of a number of
        half-planes, and that is how we represent it
        (to make it easier to check interior points) """
    def __init__(self, H):
        """ Constructs a polytope instance.

            2D vectors v satisfying Av < b are in the
            interior of the polytope.

            We use homogeneous coordinates; thus, the
            halfspace matrix H=[A|-b] encodes our constraint.

            2D (3-dimensional) homogeneous vectors v*
            satisfying H v* < 0 are in the interior of
            the polytope.

            For a nondegenerate polytope with m faces,
            A is m-by-2 and b is m-by-1, so H is m-by-3
            
            Parameters
            ---
            H : halfspace normals matrix. m-by-3 numpy array. 

            Returns
            ---
            prob : float"""
        assert H.shape[1] == 3

##        # NEVERMIND DON'T DO THE BELOW IT'S DUMB
##        # renormalize the H matrix's homogeneous transforms
##        for i in range(H.shape[0]):
##            assert np.abs(H[i,2]) != 0
##            
##            H[i,0] /= np.abs(H[i,2])
##            H[i,1] /= np.abs(H[i,2])
##            H[i,2] = np.sign(H[i,2])

        self.H = H

    def get_H(self):
        """ Get the h matrix for this polytope.

            Returns
            ---
            H : halfspace normals matrix. Numpy array m-by-3. """
        return self.H
    
    def test_point_interior(self, v, tolerance=0.0):
        """ Tests whether v lies strictly within the
            interior of this Polytope.
            
            Parameters
            ---
            v: 2D vector. numpy array, 3-by-1 (homogeneous coordinates)

            Returns
            ---
            True or False depending on whether v interior."""
        halfspace_dists = np.dot(self.H, v)
        return np.sum(halfspace_dists <= tolerance) == np.size(halfspace_dists)

    def rotated_about_origin(self, theta):
        """ Returns a new Polytope rotated by theta degrees
            about the origin.
            
            Parameters
            ---
            theta: amount (in radians) to rotate this shape
            about the origin.

            Returns
            ---
            Newly-rotated Polytope instance."""
        R = np.array([[np.cos(-theta),-np.sin(-theta), 0],
                      [np.sin(-theta), np.cos(-theta), 0],
                      [0, 0, 1]]);
        H_new = np.dot(self.get_H(), R)
        return Polytope(H_new)

    def translated_by_2D_vector(self, v):
        """ Returns a new Polytope translated by v
            
            Parameters
            ---
            v: 2D numpy array 2-by-1.

            Returns
            ---
            Newly-translated Polytope instance."""
        T = np.eye(3)
        T[0][2] = -v[0][0]
        T[1][2] = -v[1][0]
        H_new = np.dot(self.get_H(), T)
        return Polytope(H_new)

    def rotated_about_2D_point(self, theta, p):
        """ Returns a new Polytope rotated by theta degrees
            about the point p.
            
            Parameters
            ---
            theta: amount (in radians) to rotate this shape
            about the origin.

            p: 2D numpy array 2-by-1.

            Returns
            ---
            Newly-rotated Polytope instance."""
        poly_recentered = self.translated_by_2D_vector(-p)
        poly_recentered_rot = poly_recentered.rotated_about_origin(theta)
        poly_rot = poly_recentered_rot.translated_by_2D_vector(p)

        return poly_rot

    def render_polytope(self, plt, gridhalf = 100, xmax = 5.0):
        points_in = []
        points_out = []
        for x in [xmax*(i-gridhalf)/float(gridhalf) for i in range((2*gridhalf)+1)]:
            for y in [xmax*(i-gridhalf)/float(gridhalf) for i in range((2*gridhalf)+1)]:
                v = np.array([[x],[y],[1.0]])
                if self.test_point_interior(v):
                    points_in.append(v)
                else:
                    points_out.append(v)
        plt.scatter([t[0] for t in points_in],[t[1] for t in points_in])
        plt.scatter([t[0] for t in points_out],[t[1] for t in points_out])
        plt.show()


class Square(Polytope):
    """ Helper class to construct a square of a certain
        size, centered at the origin. """
    
    def __init__(self, width, height):
        """ Constructs a square of a certain
            size, centered at the origin.
            
            Parameters
            ---
            width: width of the square

            height: height of the square

            Returns
            ---
            New square polytope centered at the origin."""
        A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        b = np.array([[width],[height],[width],[height]])
        H = np.concatenate((A,-b),1)
        Polytope.__init__(self, H)

class Triangle(Polytope):
    """ Helper class to construct a triangle of a certain
        size, and radius pointing along the positive x-axis,
        centered at the origin. """
    def __init__(self, radius):
        A = np.array([[-1,0],[0.5, np.sqrt(3)/2],[0.5, -np.sqrt(3)/2]])
        inner_radius = radius * np.sin(np.pi/6)
        b = np.array([[inner_radius] for _ in range(3)])
        H = np.concatenate((A,-b),1)
        Polytope.__init__(self, H)


class Hexagon(Polytope):
    """ Helper class to construct a hexagon of a certain
        radius pointing along the positive x-axis,
        centered at the origin. """
    def __init__(self, radius):
        six_normals = [[np.cos(2.0*np.pi*(i+0.5)/6.0),np.sin(2.0*np.pi*(i+0.5)/6.0)] for i in range(6)]
        A = np.array(six_normals)
        inner_radius = radius * np.cos(np.pi/6)
        b = np.array([[inner_radius] for _ in range(6)])
        H = np.concatenate((A,-b),1)
        Polytope.__init__(self, H)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    b = np.array([[1],[1],[1],[1]])
    H = np.concatenate((A,-b),1)
    square = Polytope(H)

    print square
    assert square.test_point_interior(np.array([[.7],[.7],[1.0]]))
    assert not square.test_point_interior(np.array([[1.7],[.7],[1.0]]))
    assert not square.test_point_interior(np.array([[.7],[1.7],[1.0]]))

    print square.get_H()
    print square.rotated_about_origin(np.pi / 3).get_H()
    print square.rotated_about_origin(np.pi / 14).translated_by_2D_vector(np.array([[1.5],[1.0]])).get_H()
    
    #square.rotated_about_origin(np.pi / 14).translated_by_2D_vector(np.array([[1.5],[1.0]])).render_polytope(plt)
    #square.rotated_about_2D_point(np.pi / 12, np.array([[10.0],[0.0]])).render_polytope(plt)
    #Hexagon(2.0).rotated_about_2D_point(np.pi/2, np.array([[-4.0],[0.0]])).render_polytope(plt)
    print Hexagon(2.0).rotated_about_2D_point(np.pi/2, np.array([[-4.0],[0.0]])).get_H()

    hexagon = Hexagon(2.0)
    assert hexagon.test_point_interior(np.array([[1.9],[0.0],[1.0]]), tolerance=0.0)
    assert not hexagon.test_point_interior(np.array([[2.1],[0.0],[1.0]]), tolerance=0.0)
    assert hexagon.test_point_interior(np.array([[2.0],[0.0],[1.0]]), tolerance=0.01)
