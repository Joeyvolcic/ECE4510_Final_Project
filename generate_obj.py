#This is where we will generate the new obj file position

def write_obj_file(filename, joint1x: int, joint1y: int, joint1z: int, joint2x: int, joint2y: int, joint2z: int):
    with open(filename, 'w') as obj_file:
        obj_file.write("# OBJ file\n")

        obj_file.write("v {} {} {}\n".format(0, 0, 0))
        obj_file.write("v {} {} {}\n".format(0, 0, 10))
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y, joint1z + 10))
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y, joint1z))

        obj_file.write("v {} {} {}\n".format(10, 0, 0))
        obj_file.write("v {} {} {}\n".format(10, 0, 10))
        obj_file.write("v {} {} {}\n".format(joint1x + 10, joint1y, joint1z + 10))
        obj_file.write("v {} {} {}\n".format(joint1x + 10, joint1y, joint1z))



        # obj_file.write("v {} {} {}\n".format(joint2x, joint2y, joint2z))
        # obj_file.write("v {} {} {}\n".format(joint2x, joint2y, joint2z))
        # obj_file.write("v {} {} {}\n".format(joint2x, joint2y, joint2z))
        # obj_file.write("v {} {} {}\n".format(joint2x, joint2y, joint2z))

        obj_file.write("usemtl Material\n")
        obj_file.write("s off\n")
        obj_file.write("f 1 2 3 4 \n")
        obj_file.write("f 5 6 7 8 \n")
        #obj_file.write("f 2 3\n")

