# main.py 
# ----------------------------------------------------------------------------------------------
# Main code for lightshaper. Generates surface points for light sources and a sphere as the
# subject. Calculates the impact of each light source on each pixel of the sphere as viewed 
# through the camera
#
# Throughout this project we will use 3D polar co-ordinates r, theta, phi using mathematics 
# convention, where theta is the azimuth and phi is the polar angle. Right hand screw rule
# use for rotation convention. The angles are in degrees and distances are in meters 
#
#                             ^ z     * (x,y,z)
#                             |      / .
#                             |     .  .
#                             |phi /   .   
#                             | > .    .
#                             |  /     .
#                             | .      .
#                              /_______._________> y
#                            / .       .
#                           /    .     .
#                          /  >    .   .
#                         /  theta   . .
#                        /             . (x,y)  
#                       /                
#                      x  
# ----------------------------------------------------------------------------------------------

# import required libraries
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sys
import errno

# Define all functioins
# cart3d --------------------------------------------------------------------
# Inputs: spherical coordinates r(no unit), theta(degrees), phi(degrees)
# Output: Cartesian coordinates x, y, z, all having the same unit as input r
# --------------------------------------------------------------------------- 
def cart3d(r, theta, phi):
    theta_rad = theta * math.pi / 180
    phi_rad = phi * math.pi / 180
    x = r * math.cos(theta_rad) * math.sin(phi_rad)
    y = r * math.sin(theta_rad) * math.sin(phi_rad)
    z = math.cos(phi_rad)
    v = np.array([[x,y,z]]).T
    return v

# Define a rectangular surface
# Steps
# 1. Assume the center of the surface will pass through the origin. 
# 2. Use two orthogonal vectors, (r_1, theta_1, phi_1) and (r_2, theta_2, phi_2) to represent the plane of choice
#    along its length and width 
# 3. Derive the parameters (for later use)
# 4. Using parameters defined for the LED mat, come up with an array of coordinates for the LEDs
# 5. Translate these coordinates to the actual location of the LED mat (defined by the coordinates of its center
#    point

# Parameters of the rectangular LED mat light source
led_mat_rows = 4 # no unit. Must be an integer
led_mat_cols = 4 # no unit. Must be an integer
num_leds = led_mat_rows * led_mat_cols # number of LEDs; one LED at every intersection of a column and a row
led_mat_len = 4 # meters
led_mat_wid = led_mat_len * led_mat_rows / led_mat_cols # meters

# Individual LED parameters 
led_view_angle = 110 # degrees
led_max_output = 100 # lumens per LED

# Position parameters
r_1, theta_1, phi_1 = 1, 45, 0 # One of the vectors lying on the plane, when the plane's center is 
                               # mapped to the origin
r_2, theta_2, phi_2 = 1, 45, 90 # The second vector
mat_dist = np.array([[0.5], [0.5], [0]]) # meters. How far is the center of the mat away from the origin

# Generate LED mat surface coordinates
v1 = cart3d(r_1, theta_1, phi_1) # Convert to cartesian coordiantes
v2 = cart3d(r_2, theta_2, phi_2)
if round(np.vdot(v1, v2), 10) != 0: # If the vectors aren't orthogonal to each other..
                                    # The rounding is done because sin (cos) functions involved
                                    # converting spherical to carterisan coordinates do not return
                                    # 0 (1) when supplied with pi as the argument
    print("Vectors used to represent a plane needs to be orthogonal")
    sys.exit(errno.EDOM)

A = np.concatenate((v1.T, v2.T), axis = 0) # Form the array A as in the homogenous equation Ax = 0
U, S, Vh = la.svd(A) # Since A is 2x3 matrix, the last column of V would represent a vector in the
                     # null space of A (or, the last row of Vh transposed)
plane_params_origin = Vh[2,:] # i.e, plane parameters [a,b,c] = last row of Vh, where the plane equation is
                              # ax+by+cz = 0. Keyword '_origin' is used to indeicate that this is the version
                              # of the plane after translating its mid point to the origin
plane_params = plane_params_origin + mat_dist # Translate the plane to its actual intended locaiton

led_plane = np.empty([3, led_mat_rows * led_mat_cols]) # A collection of all the points that comprise the LED 
                                                       # plane
led_x = np.empty([led_mat_rows, led_mat_cols]) # These matrices are for visualization
led_y = np.empty([led_mat_rows, led_mat_cols]) 
led_z = np.empty([led_mat_rows, led_mat_cols])

left_top_corner = v1 * (led_mat_len/2) + v2 * (led_mat_wid/2)
col_step = v1 * led_mat_len/(led_mat_cols - 1)
row_step = v2 * led_mat_wid/(led_mat_rows - 1)
led_num = 0
for row in range(0, led_mat_rows):
    row_offset = row * row_step
    for col in range(0, led_mat_cols):
        col_offset = col * col_step
        led_plane[:,led_num] = (left_top_corner - (col_offset + row_offset))[:,0]
        led_x[row,col] = led_plane[0, led_num]
        led_y[row,col] = led_plane[1, led_num]
        led_z[row,col] = led_plane[2, led_num]
        led_num = led_num + 1

led_x = led_x + mat_dist[0,0]
led_y = led_y + mat_dist[1,0]
led_z = led_z + mat_dist[2,0]
     
# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
led_wf = ax.plot_wireframe(X = led_x, Y = led_y, Z = led_z, color = 'black')

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
