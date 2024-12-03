class FaceDataPrettyPrinter:
    def __init__(self, FD):
        self.FD = FD
    def display_hint(self):
        return 'string'
    def to_string(self):
        return "{middleVert = "+str(self.FD.referenced_value()['middleVert'][0])+\
                ", highVert = "+str(self.FD.referenced_value()['highVert'][0])+\
                ", id = "+str(self.FD.referenced_value()['id'][0])

def face_pp(val):
    if str(val.type) == 'FaceData *':
        return FaceDataPrettyPrinter(val)
    else:
        return None

gdb.pretty_printers.append(face_pp)
