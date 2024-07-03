import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from typing import List

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi

class Angle:

    def __init__(self, value, unit:str):
        assert unit in ['deg','rad'], "invalid unit"
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value} {self.unit})"

    def copy(self):
        return Angle(self.value, self.unit)

    def item(self) -> float:
        return self.value

    def in_deg(self) -> float: 
        if self.unit == 'deg':
            return self.value
        else:
            return self.value * RAD_TO_DEG

    def in_rad(self) -> float:
        if self.unit == 'rad':
            return self.value
        else:
            return self.value * DEG_TO_RAD

    def in_unit(self, unit:str) -> float:
        assert unit in ['deg', 'rad'], 'Invalid unit'
        if unit == 'deg':
            return self.in_deg()
        if unit == 'rad':
            return self.in_rad()

    def convert_to_deg(self)-> None:
        if self.unit == 'deg':
            return
        self.value = self.value * RAD_TO_DEG
        self.unit = 'deg'
        
    def convert_to_rad(self)->None:
        if self.unit == 'rad':
            return
        self.value = self.value * DEG_TO_RAD
        self.unit = 'rad'

    def convert_to(self, unit:str) -> None:
        assert unit in ['deg', 'rad'], 'Invalid unit'
        if unit == 'deg':
            self.convert_to_deg()
        if unit == 'rad':
            self.convert_to_rad()

    def assign(self, other:'Angle') -> None:
        # self gets the value from angle, but keeping its own unit
        self.value = other.in_unit(self.unit)

    def standardize(self) -> None:
        # Restrict angle between -180deg and 180deg or -pi, pi
        degrees = self.in_deg()
        degrees = degrees % 360
        if degrees > 180:
            degrees -= 360
        self.assign(Angle(degrees, 'deg'))

    def __add__(self, other:'Angle'):
        if isinstance(other, Angle):
            new = self.copy()
            new.value += other.in_unit(new.unit)
            return new
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Angle' and '{type(other).__name__}'")

    def add_in_place(self, other:'Angle'):
        self.value += other.in_unit(self.unit)

    def __sub__(self, other:'Angle') -> 'Angle':
        if isinstance(other, Angle):
            original_unit = self.unit
            diff = Angle(self.in_deg() - other.in_deg(), 'deg')
            diff.convert_to(original_unit)
            return diff
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Vector' and '{type(other).__name__}'")

    def sub_in_place(self, other:'Angle'):
        self.value -= other.in_unit(self.unit)

    def distance(self:'Angle', angle2:'Angle') -> 'Angle':
        dist = self - angle2
        dist.standardize()
        dist.value = abs(dist.value)
        return dist

    def visualize(self):

        original_unit = self.unit
        self.convert_to_rad()  # ensure angle is in radians
        
        fig, ax = plt.subplots()

        # Define the circle
        radius = 1
        circle = plt.Circle((0, 0), radius, edgecolor='b', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        # Calculate x, y coordinates for the angle
        x = radius * np.cos(self.value)
        y = radius * np.sin(self.value)

        # Draw the angle arc
        arc = Arc((0, 0), 2 * radius, 2 * radius, angle=0, theta1=0, theta2=np.degrees(self.value), color='r', linewidth=2)
        ax.add_patch(arc)

        # Draw radius lines
        plt.plot([0, radius], [0, 0], color='black', linewidth=2)  # from origin to circle edge on x-axis
        plt.plot([0, x], [0, y], color='black', linewidth=2)  # from origin to (x, y)

        # Improve annotation placement
        label_x = 1.1 * x
        label_y = 1.1 * y
        ax.annotate(f'{np.degrees(self.value):.1f}°', xy=(label_x, label_y), textcoords='offset points',
                    xytext=(10, 10), ha='center', arrowprops=dict(facecolor='black', shrink=0.05))

        # Maintain the circle and enhance plot aspect
        ax.set_aspect('equal')
        plt.xlim(-radius-1, radius+1)
        plt.ylim(-radius-1, radius+1)
        plt.title("Visual Representation")  # Optional: add a title
        plt.show()
        
        if original_unit == 'deg':
            self.convert_to_deg()

    def clip_around(self, center_angle:'Angle', size:'Angle'):
        # if the angle is in the cone, don't change it
        if self.distance(center_angle).in_deg() < size.in_deg():
            return
        # assign the angle to the closest of the two bounds
        upper_bound = center_angle + size
        lower_bound = center_angle - size
        if self.distance(upper_bound).in_deg() < self.distance(lower_bound).in_deg():
            self.assign(upper_bound)
        else:
            self.assign(lower_bound)

    def convert_to_closest_representation(self, other:'Angle') -> None:
        # finds the representation of 'self' which is max 180 deg away from 'other'
        new = self.copy()
        while (new - other).in_deg() > 180:
            new -= Angle(360, 'deg')
        while(new - other).in_deg() < -180:
            new += Angle(360, 'deg')
        self.assign(new)

class AngleVector:

    def __init__(self, values=[], unit='rad') -> None:
        assert unit in ['deg', 'rad'], 'Invalid unit'
        self.angles : List[Angle] = []
        for value in values:
            self.angles.append(Angle(value, unit))

    def unit(self):
        if len(self.angles) == 0:
            return 'rad'
        return self.angles[0].unit

    def append(self, angle:Angle):
        angle.convert_to(self.unit())
        self.angles.append(angle)

    @staticmethod
    def zeros(length) -> 'AngleVector':
        return AngleVector(np.zeros(length), 'rad')

    def __getitem__(self, index) -> Angle:
        if isinstance(index, slice):
            return AngleVector(self.angles[index.start:index.stop:index.step], self.unit())
        else:
            return self.angles[index]
    
    def __setitem__(self, index, angle:Angle) -> Angle:
        if isinstance(index, slice):
            raise(ValueError)
        else:
            self.angles[index] = angle.convert_to(self.unit())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_unit(self.unit())} {self.unit()})"

    def in_unit(self, unit) -> np.ndarray:
        assert unit in ['deg', 'rad'], 'Invalid unit'
        dim = len(self.angles)
        values = np.zeros(dim)
        for idx, angle in enumerate(self.angles):
            values[idx] = (angle.in_unit(unit))
        return values
    
    def in_deg(self) -> np.ndarray:
        dim = len(self.angles)
        values = np.zeros(dim)
        for idx, angle in enumerate(self.angles):
            values[idx] = (angle.in_deg())
        return values
    
    def in_rad(self) -> np.ndarray:
        dim = len(self.angles)
        values = np.zeros(dim)
        for idx, angle in enumerate(self.angles):
            values[idx] = (angle.in_rad())
        return values

    def convert_to(self, unit):
        assert unit in ['deg', 'rad'], 'Invalid unit'
        for angle in self.angles:
            angle.convert_to(unit)
    
    def add_in_place(self, other:'AngleVector'):
        assert len(self.angles) == len(other.angles)
        for idx, _ in enumerate(self.angles):
            self.angles[idx].add_in_place(other.angles[idx])
    
    def __add__(self, other:'AngleVector') ->'AngleVector':
        if isinstance(other, AngleVector):
            new = self.copy()
            new.add_in_place(other)
            return new
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Angle' and '{type(other).__name__}'")

    def sub_in_place(self, other:'AngleVector'):
        assert len(self.angles) == len(other.angles)
        for idx, _ in enumerate(self.angles):
            self.angles[idx].sub_in_place(other.angles[idx])

    def __sub__(self, other:'AngleVector') ->'AngleVector':
        if isinstance(other, AngleVector):
            new = self.copy()
            new.sub_in_place(other)
            return new
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Angle' and '{type(other).__name__}'")
        
    def clip_around(self, other:'AngleVector', size:Angle):
        assert len(self.angles) == len(other.angles)
        for idx, _ in enumerate(self.angles):
            self.angles[idx].clip_around(other.angles[idx], size)

    def copy(self):
        return AngleVector(self.in_deg(), 'deg')

    def visualize(self):
        figure = AngleFigure()
        for angle in self.angles:
            figure.plot(angle)
        figure.show()

    def convert_to_closest_representation(self, other:'AngleVector'):
        for i, _ in enumerate(self.angles):
            self.angles[i].convert_to_closest_representation(other.angles[i])

class AngleFigure:

    def __init__(self):
        """ Initialize the plot with a circle representing the unit circle. """
        self.fig, self.ax = plt.subplots()  # Create a figure and an axis
        self.circle = plt.Circle((0, 0), 1, edgecolor='b', facecolor='none')  # Create a circle
        self.ax.add_artist(self.circle)  # Add the circle to the axes

        # Set limits and aspect
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect('equal')  # Equal aspect ratio ensures that circle is not oval
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)

    def plot(self, angle:Angle, color='r'):
        
        angle_rad = angle.in_rad()
        angle_deg = angle.in_deg()

        """ Plot an angle on the unit circle without displaying the plot. """
        # Determine the coordinates for the end point of the angle line
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)

        # Plot the line from the center to the point on the circle
        self.ax.plot([0, x], [0, y], label=f'Angle: {angle_deg} deg', color=color)  # Line representing the angle

        # Optional: annotate the angle
        self.ax.annotate(f'{angle_deg:.2f} deg', xy=(x/2, y/2), xytext=(10, 10),
                         textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    def show(self):
        """ Display the plot with all plotted angles. """
        plt.legend()
        plt.show()