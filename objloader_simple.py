import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import generate_obj as go


class OBJ:
    def __init__(self, filename, swapyz=False): #set the swapyz to true to see the model correctly in a 3d plot
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

# Create an instance of the OBJ class with the path to your OBJ file

#obj = OBJ("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\fox.obj")

go.write_obj_file("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj", 0, 100, 0, 0, 150, 150)
obj = OBJ("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj")


# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw lines based on vertex positions
for face in obj.faces:
    vertices = obj.vertices
    x = [vertices[i - 1][0] for i in face[0]]
    y = [vertices[i - 1][1] for i in face[0]]
    z = [vertices[i - 1][2] for i in face[0]]

    # Connect the last point to the first to close the loop
    x.append(x[0])
    y.append(y[0])
    z.append(z[0])

    # Plot the line
    ax.plot(x, y, z, c='black')

# Print some debugging information
print("Vertices:", obj.vertices)
print("Faces:", obj.faces)

# Set plot limits
ax.set_xlim([-100, 100])  # Adjust the limits based on your object dimensions
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])

# Show the 3D plot
plt.show()