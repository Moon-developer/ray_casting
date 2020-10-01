import numpy as np
import matplotlib.pyplot as plt


class Settings:
    def __init__(self):
        self.width = 200
        self.height = 200

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height


class Object:
    def __init__(self):
        self.spheres = []

    def generate_spheres(self, radius: list):
        # TODO: separate spheres and plain
        self.spheres = [
            {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]),
             'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
            {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]),
             'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
            {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]),
             'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5}
        ]

    def get_spheres(self):
        return self.spheres


class RayTracer:
    def __init__(self):
        self.st = Settings()

        self.height = self.st.get_height()
        self.width = self.st.get_width()
        self.max_depth = 3
        self.camera = np.array([0, 0, 1])
        self.ratio = float(self.height) / self.st.width
        self.screen = (-1, 1 / self.ratio, 1, -1 / self.ratio)
        self.light = {
            'position': np.array([5, 5, 5]),
            'ambient': np.array([1, 1, 1]),
            'diffuse': np.array([1, 1, 1]),
            'specular': np.array([1, 1, 1])
        }

        self.objects = Object()
        self.generate_spheres()

    def generate_spheres(self):
        radius = [0.7, 0.1, 0.15]
        self.objects.generate_spheres(radius=radius)

    @staticmethod
    def normalize(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def reflected(vector, axis):
        return vector - 2 * np.dot(vector, axis) * axis

    @staticmethod
    def sphere_intersect(center, radius, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - center)
        c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

    def nearest_intersected_object(self, objects, ray_origin, ray_direction):
        distances = [self.sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
        nearest_object = None
        min_distance = np.inf
        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = objects[index]
        return nearest_object, min_distance

    def execute(self):
        image = np.zeros((self.height, self.width, 3))
        spheres = self.objects.get_spheres()
        for i, y in enumerate(np.linspace(self.screen[1], self.screen[3], self.height)):
            for j, x in enumerate(np.linspace(self.screen[0], self.screen[2], self.width)):
                pixel = np.array([x, y, 0])
                origin = self.camera
                direction = self.normalize(pixel - origin)
                color = np.zeros((3,))
                reflection = 1
                for k in range(self.max_depth):
                    nearest_object, min_distance = self.nearest_intersected_object(spheres, origin, direction)
                    if nearest_object is None:
                        break
                    intersection = origin + min_distance * direction
                    normal_to_surface = self.normalize(intersection - nearest_object['center'])
                    shifted_point = intersection + 1e-5 * normal_to_surface
                    intersection_to_light = self.normalize(self.light['position'] - shifted_point)
                    _, min_distance = self.nearest_intersected_object(spheres, shifted_point, intersection_to_light)
                    intersection_to_light_distance = np.linalg.norm(self.light['position'] - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance
                    if is_shadowed:
                        break
                    illumination = np.zeros((3,))
                    illumination += nearest_object['ambient'] * self.light['ambient']
                    illumination += nearest_object['diffuse'] * self.light['diffuse'] * np.dot(intersection_to_light,
                                                                                               normal_to_surface)
                    intersection_to_camera = self.normalize(self.camera - intersection)
                    _h = self.normalize(intersection_to_light + intersection_to_camera)
                    illumination += nearest_object['specular'] * self.light['specular'] * \
                                    np.dot(normal_to_surface, _h) ** (nearest_object['shininess'] / 4)
                    color += reflection * illumination
                    reflection *= nearest_object['reflection']
                    origin = shifted_point
                    direction = self.reflected(direction, normal_to_surface)
                image[i, j] = np.clip(color, 0, 1)
            print("%d/%d" % (i + 1, self.height))
        plt.imsave('image.png', image)


def main():
    rt = RayTracer()
    rt.execute()


if __name__ == '__main__':
    main()
