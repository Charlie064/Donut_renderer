import os
import time
import numpy as np
import dot_obj_to_points_converter as dot_obj


class Object3D:
    def __init__(self, object_size, d_object):
        self.object_size = object_size
        self.d_object = d_object
        self.points = None
        self.object_type = None
        self.funny_colour_patterns = set()



    def get_normals(self):
        X, Y, Z, u, v, _, _ = self._param_data
        return self.generate_normals(X, Y, Z, u, v)
    

    def generate_normals(self, X, Y, Z, u, v):
        du = u[1] - u[0]
        dv = v[1] - v[0]

        dX_du, dX_dv = np.gradient(X, du, dv, edge_order=2)
        dY_du, dY_dv = np.gradient(Y, du, dv, edge_order=2)
        dZ_du, dZ_dv = np.gradient(Z, du, dv, edge_order=2)

        tangent_u = np.stack([
            dX_du.flatten(), 
            dY_du.flatten(), 
            dZ_du.flatten()
        ], axis=1)
        tangent_v = np.stack([
            dX_dv.flatten(), 
            dY_dv.flatten(), 
            dZ_dv.flatten()
        ], axis=1)

        normals = np.cross(tangent_u, tangent_v)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals /= norms

        return normals
    

    def get_fun_point_colours(self):
        raise ValueError("Object has no such colour, try a solid colour like 'green'")


    def object_radius(self, points):
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return distances.max()


class Torus(Object3D):
    def __init__(self, object_size, d_object):
        super().__init__(object_size, d_object)
        self.object_type = "torus"
        self.funny_colour_patterns = {"funny_donut", "rainbow", "lifebuoy", "swedish"}
    

    def generate_meshgrid(self, num_u, num_v):
        thetas = np.linspace(0, 2*np.pi, num_u)
        phis = np.linspace(0, 2*np.pi, num_v)
        TH, PH = np.meshgrid(thetas, phis)

        X, Y, Z = self.torus_function(TH, PH)
        self._param_data = X, Y, Z, thetas, phis, TH, PH

        self.points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return self.points


    def torus_function(self, TH, PH):
        R1 = self.object_size
        R2 = 2 * R1
        X = (R2 + R1*np.cos(TH)) * np.cos(PH)
        Y = R1 * np.sin(TH)
        Z = -(R2 + R1*np.cos(TH)) * np.sin(PH)
        return X, Y, Z
    

    def get_fun_point_colours(self, selected_funny):
        TH, PH = self._param_data[5:7]

        if selected_funny == "funny_donut":
            mask = (TH <= np.pi)
            
            COLOURS = np.empty(TH.shape, dtype=object)
            COLOURS[mask] = "yellow"
            COLOURS[~mask] = "magenta"
            return COLOURS.flatten()
        
        elif selected_funny == "rainbow":
            # Rainbow donut
            colours = np.array(["red", "yellow", "green", "cyan", "blue", "magenta"], dtype=object)
            num_slices = 10

            COLOURS = np.empty(PH.shape, dtype=object)
            pie_cut_angles = np.linspace(0, 2*np.pi, num=num_slices + 1)

            for cut_i in range(len(pie_cut_angles) - 1):
                mask = (PH >= pie_cut_angles[cut_i]) & (PH <= pie_cut_angles[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]
            return COLOURS.flatten()
        
        elif selected_funny == "lifebuoy":
            colours = np.array(["white", "red"], dtype=object)

            COLOURS = np.empty(PH.shape, dtype=object)
            pie_cut_angles = np.array(
                [0, np.pi/8, 
                 np.pi/2, np.pi/2 + np.pi/8,
                 np.pi, np.pi + np.pi/8, 
                 3*np.pi/2, 3*np.pi/2 + np.pi/8,
                 2*np.pi])

            for cut_i in range(len(pie_cut_angles) - 1):
                mask = (PH >= pie_cut_angles[cut_i]) & (PH <= pie_cut_angles[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]
            return COLOURS.flatten()

        elif selected_funny == "swedish":
            colours = np.array(["yellow", "blue"], dtype=object)

            COLOURS = np.empty(PH.shape, dtype=object)

            pie_cut_angles = np.array(
                [0, np.pi/12, 
                 np.pi/2, np.pi/2 + np.pi/12,
                 np.pi, np.pi + np.pi/12, 
                 3*np.pi/2, 3*np.pi/2 + np.pi/12,
                 2*np.pi])
            
            for cut_i in range(len(pie_cut_angles) - 1):
                mask = (PH >= pie_cut_angles[cut_i]) & (PH <= pie_cut_angles[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]

            sandwich_cuts = np.array(
                [-np.pi/8, 
                 np.pi/8  
                ])

            for cut_i in range(len(sandwich_cuts) - 1):
                mask = (TH >= sandwich_cuts[cut_i]) & (TH <= sandwich_cuts[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]
            return COLOURS.flatten()    
         
        else:
            raise ValueError(f"Selected funny is not funny: {selected_funny}")


class Tetrahedron(Object3D):
    def __init__(self, object_size, d_object):
        super().__init__(object_size, d_object)
        self.object_type = "tetrahedron"
        self.funny_colour_patterns = {"rainbow"}


    def get_normals(self):
        return self.tetrahedron_normals(self.points)


    def generate_meshgrid(self, num_u=None, num_v=None):
        grid_resolution = 152
        xs = np.linspace(-self.object_size, self.object_size, grid_resolution)
        ys = np.linspace(-self.object_size, self.object_size, grid_resolution)
        zs = np.linspace(-self.object_size, self.object_size, grid_resolution)

        XS, YS, ZS = np.meshgrid(xs, ys, zs)
        X, Y, Z = self.tetrahedron_function(XS, YS, ZS)

        self.points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return self.points


    def tetrahedron_function(self, XS, YS, ZS):
        a, b, c, d = self.tetrahedron_verticies()

        P = np.stack([XS.ravel(), YS.ravel(), ZS.ravel()], axis=1)

        mask = self.points_on_tetrahedron_surface(P, a, b, c, d)

        P_surface = P[mask]
        X = P_surface[:,0]
        Y = P_surface[:,1]
        Z = P_surface[:,2]
        return X, Y, Z
        

    def points_on_tetrahedron_surface(self, P, a, b, c, d):
        margin = 1e-3
        bary = self.barycentric_coordinates(P, a, b, c, d)

        inside = np.all(bary >= -margin, axis=1) & np.all(bary <= 1 + margin, axis=1)
        on_surface = np.any(np.abs(bary) < margin, axis=1)

        return inside & on_surface


    def barycentric_coordinates(self, P, a, b, c, d):
        M = np.column_stack([a - d, b - d, c - d])
        b = (P - d).T
        solution = np.linalg.solve(M, b).T
        ALPHAS = solution[:, 0]
        BETAS  = solution[:, 1]
        GAMMAS = solution[:, 2]
        DELTAS = 1 - (ALPHAS + BETAS + GAMMAS)        
        return np.stack([ALPHAS, BETAS, GAMMAS, DELTAS], axis=1)    


    def tetrahedron_verticies(self):
        a = np.array([1, 1, 1], dtype=float)
        b = np.array([-1, -1, 1], dtype=float)
        c = np.array([-1, 1, -1], dtype=float)
        d = np.array([1, -1, -1], dtype=float)
        scale = self.object_size / np.sqrt(3)
        return scale*a, scale*b, scale*c, scale*d


    def tetrahedron_face_normals(self, a, b, c, d):
        # This order is used for indexing the normals.
        faces  = np.array([
            [b, c, d],  # opposite a
            [a, c, d],  # opposite b
            [a, b, d],  # opposite c
            [a, b, c],  # opposite d
        ])

        P = faces[:,0]
        Q = faces[:,1]
        R = faces[:,2]

        U = Q - P
        V = R - P
        normals = np.cross(U, V)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Make normal outward facing
        tetrahedron_centroid = (a + b + c + d)/4
        faces_centroids = np.mean(faces, axis=1)

        direction = faces_centroids - tetrahedron_centroid
        dot_products = np.sum(normals * direction, axis=1)
        flip = dot_products < 0
        normals[flip] *= -1

        return normals
    

    def tetrahedron_normals(self, P):
        """
        Goes through all face normals one at a time, 
        assign all point that are on the face with the face normal.
        """
        a, b, c, d = self.tetrahedron_verticies()
        face_normals = self.tetrahedron_face_normals(a, b, c, d)

        bary = self.barycentric_coordinates(P, a, b, c, d)
        normals = np.zeros_like(P)  # Each point gets a normal.

        for i in range(4):
            """When ith coordinate in bary coords (alfa, beta, gamma or delta) is ≈ 0 
            then the point has the ith normal in face_normals."""
            mask = np.abs(bary[:, i]) < 1e-3    # Find which points have coord[i] = 0

            # Assign the correct face normal to all normals selected by mask
            normals[mask] = face_normals[i]  

        return normals
    

    def get_fun_point_colours(self, selected_funny):
        if selected_funny == "rainbow":
            P = self.points
            a, b, c, d = self.tetrahedron_verticies()
            face_colours = np.array(["magenta", "green", "cyan", "red"])

            bary = self.barycentric_coordinates(P, a, b, c, d)
            points_colour = np.empty(len(P), dtype=object)

            for i in range(4):
                """When ith coordinate in bary coords (alfa, beta, gamma or delta) is ≈ 0 
                then the point has the ith normal in face_normals."""
                mask = np.abs(bary[:, i]) < 1e-3    # Find which points have coord[i] = 0

                # Assign the correct face normal to all normals selected by mask
                points_colour[mask] = face_colours[i]  

            return points_colour
        else:
            raise ValueError(f"Selected funny is not funny: {selected_funny}")


class Disk(Object3D):
    def __init__(self, object_size, d_object):
        super().__init__(object_size, d_object)
        self.object_type = "disk"
        self.funny_colour_patterns = {"rainbow"}



    def generate_meshgrid(self, num_u=100, num_v=100):
        radii = np.linspace(0, self.object_size, num_u)
        angles = np.linspace(0, 2*np.pi, num_v)
        R, TH = np.meshgrid(radii, angles)
        X, Y, Z = self.disk_function(R, TH)
        self._param_data = X, Y, Z, radii, angles, R, TH 
        
        self.points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return self.points

    
    def disk_function(self, R, TH):
        X = R * np.cos(TH)
        Y = R * np.sin(TH)
        Z = np.zeros_like(X)  # flat disk in XY plane
        return X, Y, Z
    

    def get_fun_point_colours(self, selected_funny):
        if selected_funny == "rainbow":
            colours = np.array(["red", "yellow", "green", "cyan", "blue", "magenta"], dtype=object)
            num_slices = 10
            
            R, TH = self._param_data[5:7]
            COLOURS = np.empty(TH.shape, dtype=object)
            pie_cut_angles = np.linspace(0, 2*np.pi, num=num_slices + 1)

            for cut_i in range(len(pie_cut_angles) - 1):
                mask = (TH >= pie_cut_angles[cut_i]) & (TH <= pie_cut_angles[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]
            return COLOURS.flatten()
        else:
            raise ValueError(f"Selected funny is not funny: {selected_funny}")
    


class Plane(Object3D):
    def __init__(self, object_size, d_object):
        super().__init__(object_size, d_object)
        self.object_type = "plane"
        self.funny_colour_patterns = {"rainbow"}

    

    def generate_meshgrid(self, num_u=100, num_v=100):            
        xs = np.linspace(-self.object_size, self.object_size, num_u)
        ys = np.linspace(-self.object_size, self.object_size, num_v)
        XS, YS = np.meshgrid(xs, ys)
        X, Y, Z = self.plane_function(XS, YS)
        self._param_data = X, Y, Z, xs, ys, XS, YS
        
        self.points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return self.points
    

    def plane_function(self, XS, YS):
        X = XS
        Y = YS
        Z = np.zeros_like(X)
        return X, Y, Z
    

    def get_fun_point_colours(self, selected_funny):
        if selected_funny == "rainbow":
            colours = np.array(["red", "yellow", "green", "cyan", "blue", "magenta"], dtype=object)
            num_slices = 7
            
            XS, YS = self._param_data[5:7]
            COLOURS = np.empty(XS.shape, dtype=object)
            slices = np.linspace(-self.object_size, self.object_size, num=num_slices + 1)

            for cut_i in range(len(slices) - 1):
                mask = (XS >= slices[cut_i]) & (XS <= slices[cut_i + 1])
                COLOURS[mask] = colours[cut_i % len(colours)]
            return COLOURS.flatten()
        else:
            raise ValueError(f"Selected funny is not funny: {selected_funny}")


class ImportedObject(Object3D):
    def __init__(self, object_size, d_object):
        super().__init__(object_size, d_object)
        self.object_type = "imported"



    def generate_meshgrid(self, num_u=100, num_v=100):            
        self.points = dot_obj.validated_points(file_name="skull.obj", max_points = 10000)
        
        # Center object to origin
        self.center_object = True
        if self.center_object: 
            self.move_to_origin() 

        return self.points
    

    def get_normals(self):
        self.center = self.points.mean(axis=0)
        normals = self.points - self.center

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals /= norms

        return normals
    

    def move_to_origin(self):
        old_center = self.points.mean(axis=0)
        self.points = self.points - old_center
    


class Renderer():
    def __init__(self, screen_width=None, screen_height=None, terminal_correction=0.5, object_size=5, d_object=5, object_type="torus", d_screen=None):
        self.render_luminance = True    # False: even lighting, no shadows.


        if object_type == "torus":
            self.object = Torus(object_size, d_object)
        elif object_type == "disk":
            self.object = Disk(object_size, d_object)
        elif object_type == "plane":
            self.object = Plane(object_size, d_object)
        elif object_type == "tetrahedron":
            self.object = Tetrahedron(object_size, d_object)
        elif object_type == "imported":
            self.object = ImportedObject(object_size, d_object)
        else:
            raise ValueError("Unknown object type")
        

        self.luminance_chars = ".,-~:;=!*#$@"
        self.colours = {
            "black":   "\033[30m",
            "red":     "\033[31m",
            "green":   "\033[32m",
            "yellow":  "\033[33m",
            "blue":    "\033[34m",
            "magenta": "\033[35m",
            "cyan":    "\033[36m",
            "white":   "\033[37m",
        }

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fixed_screen_size = (screen_height is not None or screen_width is not None)

        self.terminal_correction = terminal_correction

        self.fit_object_to_fov = (d_screen is None)
        self.d_screen = d_screen
        

        self.update_screen()
        self.generate_buffers()

        if self.fit_object_to_fov:
            self.d_screen = self.compute_d_screen()

        self.prev_height = None
        self.prev_width = None


    def calculate_luminance_val(self, normals):
        light_vector = np.array([0, 1, -1]).astype(float)
        light_vector /= np.linalg.norm(light_vector)

        luminance_values = np.dot(normals, light_vector)
        luminance_values = np.clip(luminance_values, 0, 1)
        return luminance_values


    def map_to_char(self, val, chars):
        val = np.clip(val, 0.0, 1.0)
        idx = int(val * (len(chars)))
        idx = min(idx, len(chars) - 1)
        return chars[idx]


    def generate_buffers(self):
        # Empty screen
        self.frame_buffer = np.full((self.screen_height, self.screen_width), fill_value=" ", dtype=object)
        # Fill depth buffer with infinity so that all points are closer and are accepted, and with the overlap counter.
        self.z_buffer = np.full((self.screen_height, self.screen_width), fill_value=np.inf)


    def update_screen(self):
        print("\033[H", end="", flush=True) # Cursor to home
        if not self.fixed_screen_size:
            terminal_dimensions = os.get_terminal_size()
            new_width = terminal_dimensions.columns
            new_height = terminal_dimensions.lines - 1  # Take into account the prompt line (-1).

            if (new_width != self.screen_width or new_height != self.screen_height):
                self.screen_width = new_width
                self.screen_height = new_height
                print("\033[2J", end="", flush=True)    # Clear terminal
                self.generate_buffers()
                if self.fit_object_to_fov:
                    self.d_screen = self.compute_d_screen()     


    def compute_d_screen(self):
        # Compute screen distance to fit object in view
        half_w = self.screen_width / 2
        half_h = (self.screen_height / 2) / self.terminal_correction
        max_radius_on_screen = min(half_w, half_h)

        obj = self.object

        # Determine max object radius
        if obj.object_type == "disk":
            r_max = obj.object_size
        elif obj.object_type == "torus":
            R1 = obj.object_size
            R2 = 2 * R1
            r_max = R2 + R1 # Outer radius
        elif obj.object_type == "plane":
            r_max = obj.object_size*1.2
        elif obj.object_type == "imported":
            points = obj.generate_meshgrid()
            r_max = obj.object_radius(points)
        else:
            r_max = obj.object_size

        return max_radius_on_screen * obj.d_object / (r_max*1.3)
    

    def resolve_colours(self, colour_appearance):
        if colour_appearance in self.object.funny_colour_patterns:
            points_colour = self.object.get_fun_point_colours(colour_appearance)
            return "funny", points_colour
        elif colour_appearance in self.colours:
            return "solid", colour_appearance
        else:
            raise ValueError("Unknown colour appearance")


    def draw_object(self):
        for point_index, (x, y, z) in enumerate(self.points):
            obj = self.object
            # Ignore point if it is inside the camera plane
            if z + obj.d_object == 0:
                continue

            # Perspective projection
            x_proj = (self.d_screen * x) / (z + obj.d_object)
            y_proj = (self.d_screen * y) / (z + obj.d_object)

            # Map to screen coordinates
            row = int(self.screen_height / 2 - y_proj * self.terminal_correction)
            col = int(self.screen_width  / 2 + x_proj)  

            if 0 <= row < self.screen_height and 0 <= col < self.screen_width:
                if z < self.z_buffer[row, col]:
                    self.z_buffer[row, col] = z
                    if self.render_luminance:
                        point_luminance = self.luminance_values[point_index]
                        point_character = self.map_to_char(point_luminance, self.luminance_chars)

                        if self.colour_mode == "funny":
                            colour_name = self.points_colour[point_index]
                        else:
                            colour_name = self.solid_colour
                        self.frame_buffer[row, col] = self.colours[colour_name] + point_character + "\033[0m"

                    else:
                        self.frame_buffer[row, col] = "@"

        for row in self.frame_buffer:
            print("".join(row))


        self.frame_buffer[:] = " "
        self.z_buffer[:] = np.inf


    def rotate_object(self, vectors, x_axis=True, y_axis=True, z_axis=True, angle_increment = np.pi/40):

        # Set angular velocity
        ax = angle_increment if x_axis else 0
        ay = angle_increment if y_axis else 0
        az = angle_increment if z_axis else 0

        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax),  np.cos(ax)]
        ])
        Ry = np.array([
            [ np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]
        ])
        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az),  np.cos(az), 0],
            [0, 0, 1]
        ])
        return (Rz @ Ry @ Rx @ vectors.T).T


    def render_animation(self, colour_appearance):
        self.points = self.object.generate_meshgrid(num_u=100, num_v=200)
        self.normals = self.object.get_normals()
        self.luminance_values = self.calculate_luminance_val(self.normals)

        # Resolve colour mode.
        mode, colour_data = self.resolve_colours(colour_appearance)
        self.colour_mode = mode
        if mode == "funny":
            self.points_colour = colour_data
        else:
            self.solid_colour = colour_data

        # Rotate tetrahedron to nicer starting position.
        if self.object.object_type == "tetrahedron":
            self.points = self.rotate_object(vectors=self.points, x_axis=True, y_axis=True, z_axis=False, angle_increment=np.pi/3)
            self.normals = self.rotate_object(vectors=self.normals, x_axis=True, y_axis=True, z_axis=False, angle_increment=np.pi/3)

        while True:
            self.update_screen()
            self.draw_object()

            # Must rotate both points and normals.
            self.points = self.rotate_object(self.points)
            self.normals = self.rotate_object(self.normals)

            if self.render_luminance:
                self.luminance_values = self.calculate_luminance_val(self.normals)
            time.sleep(0.01)


    def run(self, colour_appearance="white"):
        try:
            print("\033[?25l", end="", flush=True)  # hide cursor
            renderer.render_animation(colour_appearance) # <-- The magic happens here.
        except KeyboardInterrupt:
            pass
        finally:
            print("\033[?25h", end="", flush=True)  # show cursor



if __name__ == "__main__":
    renderer = Renderer(terminal_correction=0.5, object_size=0.001, object_type="imported")
    renderer.run(colour_appearance="green")