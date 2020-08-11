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
# cart3d -------------------------------------------------------------------
# Convert spherical to cartersian coordinates-
# Inputs: spherical coordinates r, theta(radians/degrees), phi(radians/degrees)
# Output: Cartesian coordinates x, y, z, all having the same unit as input r
# Note: Inputs can be ordinary numbers or arrays
# --------------------------------------------------------------------------- 
def cart3d(r, theta, phi, angle_unit = 'radians'):
    if angle_unit == 'degrees':
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z

# spherical3D ---------------------------------------------------------------
# Convert cartersian to spherical coordinates
# Inputs: Cartersian coordinates x, y, z
# Outputs: Spherical coordinates r (same unit as x,y,z), theta (deg/rad), 
#          phi (deg/rad)
# Note: Input can be numbers or arrays
# ---------------------------------------------------------------------------
def spherical3d(x, y, z, angle_unit = 'radians'):
    r = (x**2 + y**2 + z**2) ** 0.5
    theta = np.arctan2(y, x)
    phi = np.arctan2((x**2+y**2)**0.5, z)
    if angle_unit == 'degrees':
        theta = (theta * 180 / np.pi)
        phi = (phi * 180 / np.pi)
    return r, theta, phi

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
led_mat_rows = 2  # no unit. Must be an integer
led_mat_cols = 2  # no unit. Must be an integer
num_leds = led_mat_rows * led_mat_cols # number of LEDs; one LED at every intersection of a column and a row
led_mat_len = 1  # meters
led_mat_wid = led_mat_len * led_mat_rows / led_mat_cols # meters

# Individual LED parameters 
led_view_angle = 110 # degrees
led_max_output = 1000 # lux at 1m

# Position parameters
r_1, theta_1, phi_1 = 1, 45, 90 # One of the vectors lying on the plane, when the plane's center is
                               # mapped to the origin
r_2, theta_2, phi_2 = 1, 45, 0 # The second vector
mat_dist = np.array([[4, -4, 0]]).T # meters. Location of the center of the LED mat in 3d space

# Generate LED mat surface coordinates
x1, y1, z1 = cart3d(r_1, theta_1, phi_1, angle_unit = 'degrees') # Convert to cartesian coordiantes
x2, y2, z2 = cart3d(r_2, theta_2, phi_2, angle_unit = 'degrees')
v1 = np.array([[x1,y1,z1]]).T
v2 = np.array([[x2,y2,z2]]).T
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
plane_params = plane_params_origin + mat_dist # Translate the plane to its actual intended location

led_plane = np.empty([3, led_mat_rows * led_mat_cols]) # A collection of all the points that comprise the LED 
                                                       # plane
led_x = np.empty([led_mat_rows, led_mat_cols]) # These matrices are for visualization
led_y = np.empty([led_mat_rows, led_mat_cols]) 
led_z = np.empty([led_mat_rows, led_mat_cols])

left_top_corner = v1 * (led_mat_len/2) + v2 * (led_mat_wid/2) # Generate physical location coordinates
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
led_plane = led_plane + mat_dist


# Define the surface of the sphere
s_centre = np.array([[0,0,0]]).T # Sphere center 
s_radius = 1 # meters. Radius of the sphere
pixel_step = 45 # degrees. The granularity of theta, phi variations (lat, lon)
pix_thetas = np.arange(start = 0, stop = 360, step = pixel_step)
pix_phis = np.arange(start = 0, stop = 360, step = pixel_step)
pix_thetas, pix_phis = np.meshgrid(pix_thetas, pix_phis)
pix_x, pix_y, pix_z = cart3d(r = s_radius, theta = pix_thetas, phi = pix_phis, angle_unit = 'degrees')
pix_x = pix_x + s_centre[0,0] # Translate all points from origin reference point to the actual center of the sphere
pix_y = pix_y + s_centre[1,0]
pix_z = pix_z + s_centre[2,0]
num_pixels = pix_x.size
num_pix_rows, num_pix_cols = pix_x.shape
pixels = np.empty([3,num_pixels])
pixels = np.concatenate((pix_x.reshape([1,num_pixels]), pix_y.reshape([1,num_pixels]), pix_z.reshape([1,num_pixels])), axis = 0) # Make an array of all pixel coordinates

# Define camera parameters
cam_location = np.array([[2,0,0]]).T

# Compute the impact of every LED on pixels, one pixed at a time
pix_led_impact_mat = np.zeros([num_pixels, num_leds])
reflectivity_sigma = 45 # degrees. Surface reflectivity is modeled as a 2D gaussian with theta and phi as the variables   
for pixel in range(0, num_pixels):
    for led in range(0, led_mat_rows * led_mat_cols):
        if la.norm(led_plane[:,led]) <= s_radius: # Skip LEDs inside the sphere
            print('WARNING: LED inside sphere')
            continue
        # Find if the pixel can get any light at all from the LED in question
        pixled_ip = np.dot(pixels[:,pixel], led_plane[:,led])
        pixpix_ip = np.dot(pixels[:,pixel], pixels[:,pixel])
        if pixled_ip > pixpix_ip: # This means the light from the LED in question cannot reach this pixel,
                                        # because the pixel is on the "back side" of the sphere w.r.to this LED
            #print(str(pixels[:,pixel]), ' not reachable to LED ', str(led_plane[:,led]))
            continue
        # Find the rotation matrix to be used to find new coordinates for vectors after rotating the axes to have z-axis 
        # align with the vector representing the pixel point we are looking at in this iteration 
        rot_theta = -pix_thetas[int(pixel/num_pix_cols), pixel%num_pix_cols] # Negative sign because we need to rotate all vectors
        rot_phi = -pix_phis[int(pixel/num_pix_cols), pixel%num_pix_cols]  # clockwise by this theta and phi
        rotation_mat1 = np.array([[np.cos(rot_theta), -np.sin(rot_theta), 0],[np.sin(rot_theta), np.cos(rot_theta), 0],[0, 0, 1]]) # Rotation around z axis
        rotation_mat2 = np.array([[np.cos(rot_phi), 0, np.sin(rot_phi)], [0, 1, 0], [-np.sin(rot_phi), 0, np.cos(rot_phi)]]) # Rotation around y axis
        # Translate and rotate the incident light line and cam-pixel line
        translated_led_loc  = led_plane[:,led:led+1] - pixels[:,pixel:pixel+1] # First we translate the origin to convert the incident light line to a vector
        translated_cam_loc = cam_location - pixels[:,pixel:pixel+1]
        rotated_inc_vector = rotation_mat2 @ rotation_mat1 @ translated_led_loc # Then we rotate the incident light vector
        rotated_cam_vector = rotation_mat2 @ rotation_mat1 @ translated_cam_loc
        # Now find the theta and phi of the incident light and camera-pix line on this bases system            
        inc_r, inc_theta, inc_phi = spherical3d(rotated_inc_vector[0,0], rotated_inc_vector[1,0], rotated_inc_vector[2,0], angle_unit = 'degrees')
        cam_r, cam_theta, cam_phi = spherical3d(rotated_cam_vector[0,0], rotated_cam_vector[1,0], rotated_cam_vector[2,0], angle_unit = 'degrees')
        # Calculate the impact of the led in question on the pixel in question (impact accounts for reflected light and attenuation of incident and reflected lights
        reflectivity_mu_theta = inc_theta + 180 # degrees. Incident light gets reflected primarily at incident azimuth+180 
        reflectivity_mu_phi = inc_phi # degrees. As far as polar angle is concerned, it doesn't change, as the angle of reflection = angle of incidence
        reflection_factor = (1/(2*np.pi*reflectivity_sigma**2)) * np.exp(-0.5 * ((cam_theta - reflectivity_mu_theta)**2+(cam_phi - reflectivity_mu_phi)**2) / reflectivity_sigma**2) 
        inc_attenuation = 1/(la.norm(led_plane[:,led] - pixels[:,pixel])**2) 
        cam_attenuation = 1/(la.norm(cam_location - pixels[:,pixel])**2)
        pix_led_impact_mat[pixel, led] = cam_attenuation * reflection_factor * inc_attenuation

led_brightness = np.ones([num_leds, 1])*led_max_output # lux value of LEDs in the mat
pix_brightness = pix_led_impact_mat @ led_brightness 

## Visualization
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d') 
led_wf = ax1.plot_wireframe(X = led_x, Y = led_y, Z = led_z, color = 'black')
sphere_wf = ax1.plot_wireframe(X = pix_x, Y = pix_y, Z = pix_z, color = 'blue')

# Customize the z axis.
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_zlim(-3, 3)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
Xn = np.arange(0,num_pixels,1)
ax2.plot(Xn, pix_brightness.flatten())

plt.show()
