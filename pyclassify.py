# Limit point outputs
MAX_DISPLAY = 10
# TARGET FILE TO PROCESS
FILE = 'test_classes.out'

# Usage: python3 pyclassify.py

# Name mapping to class IDs
#              0   1     2     3         4
class_names = ['','min','max','regular','saddle']
insanity, classes = list(), dict((k,list()) for k in range(len(class_names)))
with open(FILE,'r') as f:
    for l in f.readlines():
        # Longer form output detected
        if "Upper:" in l and "Lower:" in l:
            # Other output format
            vertex_id = int(l.split(' ',3)[2])
            for last_int, class_name in enumerate(class_names[1:]):
                if class_name in l:
                    classes[last_int+1].append(vertex_id)
                    print(f"Point {vertex_id} is {class_names[last_int+1]}")
                    break
            # Skip normal processing
            continue
        # Insanity or short form output detected
        last_int = int(l.rstrip().rsplit(' ',1)[1])
        if 'INSANITY' in l:
            insanity.append(last_int)
        else:
            vertex_id = int(l.rsplit(" ",3)[1])
            classes[last_int].append(vertex_id)
            print(f"Point {vertex_id} is {class_names[last_int]}")

print(f"{len(insanity)} insane points detected: {insanity[:MAX_DISPLAY]}")
for name, clinfo in zip(class_names, classes.values()):
    if name == "":
        continue
    print(f"{len(clinfo)} {name} points detected: {clinfo[:MAX_DISPLAY]}")

