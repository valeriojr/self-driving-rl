import numpy


def load_obj(path):
    vertices = []
    normals = []
    texture_coordinates = []
    buffer = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            elif line.startswith('v '):
                vertices.append(numpy.array(list(map(float, line.split()[1:]))))
            elif line.startswith('vn '):
                normals.append(numpy.array(list(map(float, line.split()[1:]))))
            elif line.startswith('vt '):
                texture_coordinates.append(numpy.array(list(map(float, line.split()[1:]))))
            elif line.startswith('f '):
                indices = line.split()[1:]
                if len(indices) != 3:
                    raise ValueError()
                for index in indices:
                    v, vt, vn = index.split('/')
                    buffer.extend(vertices[int(v) - 1])
                    # buffer.extend(normals[int(vn) - 1])
                    buffer.extend(texture_coordinates[int(vt) - 1])

    return numpy.array(buffer, dtype=numpy.float32)
