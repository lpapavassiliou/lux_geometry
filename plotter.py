import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from lux_geometry.point3D import Point3D
from lux_geometry.rotations import HomogeneusMatrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Visual3D:
    def __init__(self) -> None: # Init method creates a new figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.frame_scaling = 1.0
        self.force_scaling = 0.02

    def plot_point(self, point):
        self.ax.scatter(point[0], point[1], point[2], c='r', marker='o')

    def plot_points(self, points):
        for point in points:
            self.plot_point(point)

    def plot_line(self, point1, point2, color=None, linewidth=1.0):

        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        z_values = [point1[2], point2[2]]
        
        # Plot the line between the two points
        self.ax.plot(x_values, y_values, z_values, marker='o', color=color, linewidth=linewidth)
    
    def plot_lines(self, lines, color=None, linewidth=1.0):
        for line in lines:
            self.plot_line(*line, color=color, linewidth=linewidth)

    def plot_vector(self, origin, end, color=None, linewidth=None):
        """
        Plots a 3D vector as an arrow.

        Parameters:
        origin: The starting point of the vector [x, y, z].
        end: The end point of the vector [x, y, z].
        """

        # Extract the components of the origin and calculate the vector components
        x, y, z = origin
        u, v, w = [end[i] - origin[i] for i in range(3)]

        lenght = np.linalg.norm(origin - end)

        # Plot the vector as an arrow
        self.ax.quiver(x, y, z, u, v, w, length=lenght, color=color, linewidth=linewidth, normalize=True)

    def plot_vectors(self, vectors, color=None, linewidth=None):
        for vector in vectors:
            self.plot_vector(*vector, color=color,linewidth=linewidth)

    @staticmethod
    def build_frame(World_T_Frame:HomogeneusMatrix, scale=1.0):
        World_newOrigin = World_T_Frame.get_t()
        R = scale * World_T_Frame.get_R()
        World_newXAxisEnd = R[:,0] + World_newOrigin
        World_newYAxisEnd = R[:,1] + World_newOrigin
        World_newZAxisEnd = R[:,2] + World_newOrigin
        lines = [[World_newOrigin, World_newXAxisEnd],
                [World_newOrigin, World_newYAxisEnd],
                [World_newOrigin, World_newZAxisEnd]]
        return lines

    def plot_frame(self, World_T_Frame : HomogeneusMatrix, color=None):
        lines_Frame = self.build_frame(World_T_Frame, scale=self.frame_scaling)
        self.plot_vectors(lines_Frame, color=color)

    def set_ax_limits(self, dim, offset=[0, 0, 0]):
        x_lim = [-dim + offset[0], dim + offset[0]]
        y_lim = [-dim + offset[1], dim + offset[1]]
        z_lim = [-dim + offset[2], dim + offset[2]]
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_zlim(z_lim)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Scene')

    def show(self, dim=None, offset=[0, 0, 0]):
        if dim is not None:
            x_lim = [-dim + offset[0], dim + offset[0]]
            y_lim = [-dim + offset[1], dim + offset[1]]
            z_lim = [-dim + offset[2], dim + offset[2]]
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
            self.ax.set_zlim(z_lim)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Scene')
        plt.show()

    def show_in_html(self, dim=None, offset=[0, 0, 0]):
        if dim is not None:
            x_lim = [-dim + offset[0], dim + offset[0]]
            y_lim = [-dim + offset[1], dim + offset[1]]
            z_lim = [-dim + offset[2], dim + offset[2]]
            self.ax.set_xlim(x_lim)
            self.ax.set_ylim(y_lim)
            self.ax.set_zlim(z_lim)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Scene')
        
        import mpld3
        import os
        
        # Save the plot to an HTML file
        html_str = mpld3.fig_to_html(plt.gcf())
        file_name = 'plot.html'
        with open(file_name, 'w') as f:
            f.write(html_str)
        
        # Print the file path so you can open it manually
        filepath = os.path.abspath(file_name)
        print(f'Plot saved to: file://{filepath}')

    def clear(self):
        self.ax.cla()
    
    def plot_force(self, Frame_forceLocation : Point3D, Frame_forceVector, World_T_Frame : HomogeneusMatrix = None, linewidth=3):
        if World_T_Frame is None:
            World_T_Frame = HomogeneusMatrix()
        World_forceOrigin = World_T_Frame.transform(Frame_forceLocation)
        World_forceEnd = World_T_Frame.transform(Frame_forceLocation + self.force_scaling * Frame_forceVector)
        self.plot_vector(World_forceOrigin, World_forceEnd, linewidth=linewidth)
    
    def plot_box(self, points):
        # Define the 6 faces of the box using the points
        faces = [
            [points[0], points[1], points[2], points[3]],  # Bottom face
            [points[4], points[5], points[6], points[7]],  # Top face
            [points[0], points[1], points[5], points[4]],  # Front face
            [points[2], points[3], points[7], points[6]],  # Back face
            [points[1], points[2], points[6], points[5]],  # Right face
            [points[4], points[7], points[3], points[0]]   # Left face
        ]
        # Plot the faces
        self.ax.add_collection3d(Poly3DCollection(faces, facecolors='black', linewidths=1, edgecolors='black', alpha=.25))

    @staticmethod
    def update_figure():
        plt.draw()