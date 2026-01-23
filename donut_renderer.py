import os
import time
import numpy as np


class Renderer():
    def __init__(self, screen_width=50, screen_height=50, terminal_correction=0.5, d_screen=20):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.terminal_correction = terminal_correction
        self.d_screen = d_screen    # Distance from screen

        self.object_types = ["circle"]

        # Empty screen
        self.screen_pixels = np.full((screen_height, screen_width), fill_value=" ")

        # Fill depth buffer with infinity so that all points are closer and are accepted, and with the overlap counter.
        self.depth_buffer = np.full((screen_height, screen_width), np.inf)

        # All generated points
        self.points = None


    
    def generate_circle_points(self):
        radius = 5
        z = 10
        num_points = 100
        radii = np.linspace(0, radius, num_points)
        thetas = np.linspace(0, 2*np.pi, num_points)
        radius_points = []  # start empty
        for r in radii:
            points = np.stack([
                r * np.cos(thetas),
                r * np.sin(thetas),
                np.full(num_points, z)
            ], axis=1)
            radius_points.append(points)  # Store cirle array
        # Combine all circles into a single array
        self.points = np.vstack(radius_points)
        return self.points


    def draw_object(self):
        self.screen_pixels[:] = " "
        self.depth_buffer[:] = np.inf

        for x, y, z in self.points:
            # Point is inside camera plane
            if z == 0:
                continue

            # Perspective projection
            x_proj = (self.d_screen * x) / z
            y_proj = (self.d_screen * y) / z

            # Map to screen coordinates
            row = int((self.screen_height // 2 - y_proj)*self.terminal_correction)
            col = int(self.screen_width  // 2 + x_proj)

            if 0 <= row < self.screen_height and 0 <= col < self.screen_width:
                if z < self.depth_buffer[row, col]:
                    self.depth_buffer[row, col] = z
                    self.screen_pixels[row, col] = "@"

        for row in self.screen_pixels:
            print("".join(row))



    def rotate_object(self, x_axis=True, y_axis=False, z_axis=False, angle_increment = np.pi/20):
        sin_phi = np.sin(angle_increment)
        cos_phi = np.cos(angle_increment)

        if x_axis:
            Rx = np.array([[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]])
            self.points = (Rx @ self.points.T).T

            #for i, point in enumerate(self.points):
                #self.points[i] = Rx @ point

        if y_axis:
            Ry = np.array([[cos_phi, 0, sin_phi], [0, 1, 0], [-sin_phi, 0, cos_phi]])
            self.points = (Ry @ self.points.T).T


        if z_axis:
            Rz = np.array([[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]])  
            self.points = (Rz @ self.points.T).T



    def render_animation(self):
        self.generate_circle_points()
        while True:
            os.system("cls" if os.name=="nt" else "clear")  # clear terminal
            self.draw_object()
            self.rotate_object(x_axis=False, y_axis=False, z_axis=False)
            time.sleep(0.05)



if __name__ == "__main__":
    renderer = Renderer()
    renderer.render_animation()

    












