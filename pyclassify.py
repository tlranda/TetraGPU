class_names = ['','max','min','regular','saddle']
insanity, classes = list(), dict((k,list()) for k in range(len(class_names)))
with open('test_classes.out','r') as f:
    for l in f.readlines():
        last_int = int(l.rstrip().rsplit(' ',1)[1])
        if 'INSANITY' in l:
            insanity.append(last_int)
        else:
            vertex_id = int(l.rsplit(" ",3)[1])
            classes[last_int].append(vertex_id)
            print(f"Point {vertex_id} is {class_names[last_int]}")

print(f"{len(insanity)} insane points detected")
for name, clinfo in zip(class_names, classes.values()):
    if name == "":
        continue
    print(f"{len(clinfo)} {name} points detected")

