import matplotlib.pyplot as plt
import pdb
# Load and parse data
tetras = []
vertices = set()
with open('vertices.txt', 'r') as f:
    for line in f.readlines():
        tetras.append(list(map(int,
                               line.rstrip().split(':')[1][2:-2].split(' '))))
        vertices = vertices.union(set(tetras[-1]))
vertices = sorted(vertices)

faces = dict()
with open('faces.txt', 'r') as f:
    for line in f.readlines():
        get_face, get_vertices = line.rstrip().split('[')[1:]
        face_id = int(get_face.split(']')[0])
        face_vertices = list(map(int,
                                 get_vertices[:-1].split(', ')))
        faces[face_id] = face_vertices
faces = list(faces.values())
first_face = []
for vertex in vertices:
    is_found = False
    for found, face in enumerate(faces):
        if face[0] == vertex:
            first_face.append(found)
            is_found = True
            break
    if not is_found:
        first_face.append(len(faces))
print("First faces occur at:", first_face)

# Demonstrate sorting
for idx, val in enumerate(tetras):
    print("tetra", idx, val)
    assert (val[0] < val[1]) and (val[1] < val[2]) and (val[2] < val[3])
for idx, val in enumerate(faces):
    print("face", idx, val)
    assert (val[0] < val[1]) and (val[1] < val[2])

# Kernel algorithm one-thread-at-a-time
outputs = []
missing = 0
search_length = []
for TID in range(len(tetras)*4):
    tetraID = int(TID / 4)
    vertexID = TID % 4
    # Warp shuffles allow any thread to know ALL existing vertices
    looked_up_vertices = tetras[tetraID]
    # Tetras specification guarantees that vertex IDs are strictly increasing
    face_to_look_for = [looked_up_vertices[0]] if vertexID != 1 else [looked_up_vertices[1]]
    face_to_look_for.append(looked_up_vertices[2] if (vertexID == 1) or (vertexID == 2) else looked_up_vertices[1])
    face_to_look_for.append(looked_up_vertices[3] if vertexID != 0 else looked_up_vertices[2])
    # Find the faceID
    found_face_id = -1
    attempt_slice = faces[first_face[face_to_look_for[0]]:first_face[face_to_look_for[0]+1]]
    for idx, face in enumerate(attempt_slice):
        if (face[1] == face_to_look_for[1]) and (face[2] == face_to_look_for[2]):
            found_face_id = first_face[face_to_look_for[0]]+idx
            break
    if found_face_id < 0:
        missing += 1
    search_length.append(idx)
    # Write in order by appending
    outputs.append(found_face_id)
if missing > 0:
    print(f"FAILED TO FIND {missing} FACES")
    exit()

fig, ax = plt.subplots()
ax.scatter(range(len(search_length)), search_length)
ax.set_xlabel("TID")
ax.set_ylabel("Scan length until thread's face ID is located")
ax.set_title("Per-thread scan lengths as divergence check")
fig2, ax2 = plt.subplots()
#counts = sorted(set(search_length))
counts = range(min(search_length), max(search_length))
per_count = [search_length.count(count) for count in counts]
ax2.plot(range(len(counts)), per_count, marker='.')
ax2.set_xlabel("Scan length")
ax2.set_ylabel("Times Occurred across all threads")
ax2.set_title("Occurrence count of scan lengths")
fig3, ax3 = plt.subplots()
cdf = [0]+[sum(per_count[:idx])/sum(per_count) for idx in range(1, len(per_count))]
ax3.plot(range(len(cdf)), cdf, marker='.')
ax3.set_xlabel("Scan length")
ax3.set_ylabel("CDF")
ax3.set_title("CDF of scan lengths")
plt.show()

for tetraID in range(len(tetras)):
    for faceID in range(4):
        print(f"tetra_{tetraID}", f"face_{faceID} =", outputs[(tetraID*4) + faceID])

