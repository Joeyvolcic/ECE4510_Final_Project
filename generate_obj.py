import math
#This is where we will generate the new obj file position

def write_obj_file(filename, joint1x: int, joint1y: int, joint1z, joint2y: int, joint2z: int):
    #scale factor
    offset : int = 10 #we might want to pass this as an argument into the function
    joint_length : int = math.sqrt((joint1x - joint1x)**2 + (joint2y - joint1y)**2 + (joint2z - joint1z)**2)
    #calculates the points so that the side lengths of the arms are alwayts equal
    if joint1y != 0:
        arm1_angle = math.atan((joint1z) / (joint1y))
    else:
        arm1_angle = 0
    
    if joint2y - joint1y != 0:
        arm2_angle = math.atan((joint2z - joint1z) / (joint2y - joint1y))
    else:
        arm2_angle = 0
    arm1_Zoff = offset * math.cos(arm1_angle)
    arm1_Yoff = offset * math.sin(arm1_angle)
    arm2_Yoff = offset * math.sin(arm2_angle)
    arm2_Zoff = offset * math.cos(arm2_angle)
    


    
    with open(filename, 'w') as obj_file:
        obj_file.write("# OBJ file\n")

        # we dont need to write so many vertacies, a lot of them are the same
        obj_file.write("#First Segment\n")
        obj_file.write("#the first segment can only move in the y direction\n")
        obj_file.write("#coordinate format x y z\n")
        obj_file.write("v {} {} {}\n".format(0, offset * math.sin(arm1_angle), arm1_Zoff))
        obj_file.write("v {} {} {}\n".format(0, 0, offset))
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y, joint1z + offset))
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y + offset * math.sin(arm1_angle), joint1z + arm1_Zoff))

        #The end of the joins should alway create a 10 by 10 square its should also always be at a 90 degree angle

        obj_file.write("\n")
        obj_file.write("v {} {} {}\n".format(offset, offset * math.sin(arm1_angle),arm1_Zoff))
        obj_file.write("v {} {} {}\n".format(offset, 0, offset))
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y , joint1z + offset))
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y + offset * math.sin(arm1_angle), joint1z + arm1_Zoff))


        # obj_file.write("\n")
        # obj_file.write("v {} {} {}\n".format(offset, 0, 0))
        # obj_file.write("v {} {} {}\n".format(offset, 0, offset))
        # obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y, joint1z + offset))
        # obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y, joint1z))


        obj_file.write("\n")
        obj_file.write("#Second Segment\n")
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y + arm1_Yoff, joint1z + arm1_Zoff))
        obj_file.write("v {} {} {}\n".format(joint1x, joint1y - arm2_Yoff + arm1_Yoff, joint1z + arm1_Zoff + arm2_Zoff))   
        obj_file.write("v {} {} {}\n".format(joint1x, joint2y + offset - arm2_Yoff + arm1_Yoff, joint2z + offset + arm1_Zoff + arm2_Zoff))
        obj_file.write("v {} {} {}\n".format(joint1x, joint2y - arm2_Yoff + arm1_Yoff, joint2z+ arm1_Zoff + arm2_Zoff))

        obj_file.write("\n")
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y + arm1_Yoff, joint1z + arm1_Zoff))
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint1y - arm2_Yoff + arm1_Yoff, joint1z + arm1_Zoff + arm2_Zoff))
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint2y + offset - arm2_Yoff + arm1_Yoff, joint2z + offset + arm1_Zoff + arm2_Zoff))
        obj_file.write("v {} {} {}\n".format(joint1x + offset, joint2y - arm2_Yoff + arm1_Yoff, joint2z+ arm1_Zoff + arm2_Zoff))


        # obj_file.write("v {} {} {}\n".format(joint1x - offset, joint1y, joint1z))
        # obj_file.write("v {} {} {}\n".format(joint1x - offset, joint1y, joint1z + offset))
        # obj_file.write("v {} {} {}\n".format(joint1x - offset, joint2y, joint2z + offset))
        # obj_file.write("v {} {} {}\n".format(joint1x - offset, joint2y, joint2z))

        obj_file.write("\n")
        # obj_file.write("v {} {} {}\n".format(joint1x , joint1y, joint1z))
        # obj_file.write("v {} {} {}\n".format(joint1x , joint1y + offset, joint1z + offset))
        # obj_file.write("v {} {} {}\n".format(joint1x , joint2y + added_length2, joint2z  + offset))
        # obj_file.write("v {} {} {}\n".format(joint1x , joint2y + added_length2, joint2z  + offset))

        obj_file.write("#Base\n")
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), -(joint_length / 10), 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), -(joint_length / 10), 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), -(joint_length / 10), 0 + (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), -(joint_length / 10), 0 + (joint_length / 4)))
      
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), 0, 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), 0, 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), 0, 0 + (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), 0, 0 + (joint_length / 4)))

        # obj_file.write("#Joint Buldge\n")
        # obj_file.write("v {} {} {}\n".format(joint1x - (joint_length / 10), joint1y + (joint_length / 5), joint1z - (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 5), joint1z - (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 5), joint1z + (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x- (joint_length / 10), joint1y + (joint_length / 5), joint1z + (joint_length / 10)))

        # obj_file.write("v {} {} {}\n".format(joint1x - (joint_length / 10), joint1y, joint1z - (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y, joint1z - (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y, joint1z + (joint_length / 10)))
        # obj_file.write("v {} {} {}\n".format(joint1x- (joint_length / 10), joint1y, joint1z + (joint_length / 10)))



        obj_file.write("usemtl Material\n")
        obj_file.write("s off\n")

        obj_file.write("\n")
        obj_file.write("#Faces\n")

        obj_file.write("#Base\n")
        obj_file.write("f 17 18 19 20 #212121\n")
        obj_file.write("f 21 22 23 24 #212121\n")
        obj_file.write("f 17 18 22 21 #212121\n")
        obj_file.write("f 18 19 23 22 #212121\n")
        obj_file.write("f 19 20 24 23 #212121\n")
        obj_file.write("f 20 17 21 24 #212121\n")
     
        obj_file.write("#Arm 1\n")
        obj_file.write("f 1 2 3 4 #f25c19\n")
        obj_file.write("f 5 6 7 8 #f25c19\n")
        obj_file.write("f 1 2 6 5 #f25c19\n")
        obj_file.write("f 2 3 7 6 #f25c19\n")
        obj_file.write("f 3 4 8 7 #f25c19\n")

        obj_file.write("#Arm 2\n")
        obj_file.write("f 9 10 11 12 #f25c19\n")
        obj_file.write("f 13 14 15 16 #f25c19\n")
        obj_file.write("f 9 10 14 13 #f25c19\n")
        obj_file.write("f 10 11 15 14 #f25c19\n")
        obj_file.write("f 11 12 16 15 #f25c19\n")



        # obj_file.write("f 9 10 11 12 #212121\n")
        # obj_file.write("f 13 14 15 16 #212121\n")
        # obj_file.write("f 9 10 14 13 #212121\n")
        # obj_file.write("f 10 11 15 14 #212121\n")
        # obj_file.write("f 11 12 16 15 #212121\n")

        # obj_file.write("#Joint Buldge\n")
        # obj_file.write("f 25 26 27 28 #212121\n")
        # obj_file.write("f 29 30 31 32 #212121\n")
        # obj_file.write("f 25 26 30 29 #212121\n")
        # obj_file.write("f 26 27 31 30 #212121\n")
        # obj_file.write("f 27 28 32 31 #212121\n")
        # obj_file.write("f 28 25 29 32 #212121\n")




def generate_base_obj(filename, joint1x: int, joint1y: int, joint1z: int, joint2x: int, joint2y: int, joint2z: int):
    joint_length : int = math.sqrt((joint2x - joint1x)**2 + (joint2y - joint1y)**2 + (joint2z - joint1z)**2)
    with open(filename, 'w') as obj_file:

        obj_file.write("#Base\n")
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), (joint_length / 10), 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), (joint_length / 10), 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), (joint_length / 10), 0 + (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), (joint_length / 10), 0 + (joint_length / 4)))

        obj_file.write("#Base\n")
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), 0, 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), 0, 0 - (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 + (joint_length / 4), 0, 0 + (joint_length / 4)))
        obj_file.write("v {} {} {}\n".format(0 - (joint_length / 4), 0, 0 + (joint_length / 4)))

        obj_file.write("#Joint Buldge\n")
        obj_file.write("v {} {} {}\n".format(joint1x - (joint_length / 10), joint1y + (joint_length / 10), joint1z - (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 10), joint1z - (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 10), joint1z + (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x- (joint_length / 10), joint1y + (joint_length / 10), joint1z + (joint_length / 10)))

        obj_file.write("#Joint Buldge top\n")
        obj_file.write("v {} {} {}\n".format(joint1x - (joint_length / 10), joint1y + (joint_length / 10) + (joint_length / 5), joint1z - (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 10) + (joint_length / 5), joint1z - (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x + (joint_length / 10), joint1y + (joint_length / 10) + (joint_length / 5), joint1z + (joint_length / 10)))
        obj_file.write("v {} {} {}\n".format(joint1x- (joint_length / 10), joint1y + (joint_length / 10) + (joint_length / 5), joint1z + (joint_length / 10)))

        obj_file.write("\n")
        obj_file.write("#Faces\n")
        obj_file.write("f 1 2 3 4 #123123\n")
        obj_file.write("f 5 6 7 8 #212121\n")
        obj_file.write("f 1 2 6 5 #212121\n")
        obj_file.write("f 2 3 7 6 #212121\n")
        obj_file.write("f 3 4 8 7 #212121\n")
        obj_file.write("f 4 1 5 8 #212121\n")
        obj_file.write("f 9 10 11 12 #faedd2\n")
        obj_file.write("f 13 14 15 16 #faedd2\n")
        obj_file.write("f 9 10 14 13 #faedd2\n")
        obj_file.write("f 10 11 15 14 #faedd2\n")
        obj_file.write("f 11 12 16 15 #faedd2\n")
        obj_file.write("f 12 9 13 16 #faedd2\n")
        




    


