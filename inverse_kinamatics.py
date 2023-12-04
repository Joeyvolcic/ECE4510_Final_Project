import math
import numpy as np

def inverse_kinematics(px,pz, l1, l2):
    # Calculate the distance from the origin to the end effector
    distance = np.sqrt(px**2 + pz**2)

    # Check if the desired position is within the robot's reach
    if distance > l1 + l2 or distance < np.abs(l1 - l2):
        print("Desired position is out of reach")
        return None

    # Calculate the angles using inverse trigonometry
    theta2 = np.arccos((px**2 + pz**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta1 = np.arctan2(pz, px) - np.arctan2(l2* np.sin(theta2), l1 + l2 * np.cos(theta2))*180/np.pi

    return theta1, theta2

def rotateZ(theta):
    rz = np.array([[math.cos(theta), - math.sin(theta), 0, 0],
                   [math.sin(theta), math.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return rz

def translate(dx, dy, dz):
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    return t

def FK(angle, link):
    n_links = len(link)
    P = []
    P.append(np.eye(4))
    for i in range(0, n_links):
        R = rotateZ(angle[i]/180*math.pi)
        T = translate(link[i], 0, 0)
        P.append(P[-1].dot(R).dot(T))
    return P

def IK(target, angle, link, max_iter = 10000, err_min = 0.1):
    solved = False
    err_end_to_target = math.inf
    
    for loop in range(max_iter):
        for i in range(len(link)-1, -1, -1):
            P = FK(angle, link)
            end_to_target = target - P[-1][:3, 3]
            err_end_to_target = math.sqrt(end_to_target[0] ** 2 + end_to_target[1] ** 2)
            if err_end_to_target < err_min:
                solved = True
            else:
                # Calculate distance between i-joint position to end effector position
                # P[i] is position of current joint
                # P[-1] is position of end effector
                cur_to_end = P[-1][:3, 3] - P[i][:3, 3]
                cur_to_end_mag = math.sqrt(cur_to_end[0] ** 2 + cur_to_end[1] ** 2)
                cur_to_target = target - P[i][:3, 3]
                cur_to_target_mag = math.sqrt(cur_to_target[0] ** 2 + cur_to_target[1] ** 2)

                end_target_mag = cur_to_end_mag * cur_to_target_mag

                if end_target_mag <= 0.0001:    
                    cos_rot_ang = 1
                    sin_rot_ang = 0
                else:
                    cos_rot_ang = (cur_to_end[0] * cur_to_target[0] + cur_to_end[1] * cur_to_target[1]) / end_target_mag
                    sin_rot_ang = (cur_to_end[0] * cur_to_target[1] - cur_to_end[1] * cur_to_target[0]) / end_target_mag

                rot_ang = math.acos(max(-1, min(1,cos_rot_ang)))

                if sin_rot_ang < 0.0:
                    rot_ang = -rot_ang

                # Update current joint angle values
                angle[i] = angle[i] + (rot_ang * 180 / math.pi)

                if angle[i] >= 360:
                    angle[i] = angle[i] - 360
                if angle[i] < 0:
                    angle[i] = 360 + angle[i]
                  
        if solved:
            break
            
    return angle, err_end_to_target, solved, loop

def get_path(positions: list, thetaZ, ids): # position  = [[x1,y1],[x2,y2]]
    joint_length = 100
    theta1 = 0
    theta2 = 0
    path= []
    
    if len(positions[0]) != 2 and len(positions[1]) != 2:
        return [90, 90, 0]

    thetaZ = thetaZ + 180
    print("thetaZ",thetaZ)
    #Calculates the theta of the first x,y point
    print("1 positions", positions[0][1], positions[0][0])

    if ids[0][0] == 1: 
        theta1 = math.degrees(np.arctan2(positions[0][0], positions[0][1]))
        theta2 = int(math.degrees(np.arctan2(positions[1][0], positions[1][1])))
    else:
        theta1 = int(math.degrees(np.arctan2(positions[1][0], positions[1][1])))
        theta2 = math.degrees(np.arctan2(positions[0][0], positions[0][1]))

    

    # theta1 = theta1/310*360
    theta1 = int((theta1 - thetaZ + 360) % 360)
    if theta1 < 0:
        theta1 = theta1 + 180
    
     
    print("theta1 normalized: ", theta1)
    
    # Calculates the theta of the second x,y point

    # theta2 = theta2/310*360
    theta2 = int((theta2 - thetaZ) % 360)
    if theta2 < 0:
        theta2 = theta2 + 0
    
    print("theta2 normalized: ", (theta2) % 360)

    #first rotation
    print("rotation1")
    if theta1 > 180:
        for i in range(180, theta1):
            path.append([90, 90, i])
    else:
        for i in range(180, theta1, -1):
            path.append([90, 90, i])
        
    distance = math.sqrt(positions[0][0]**2 + positions[0][1]**2)

    angle, err, solved, iteration = IK([-distance,0,0], [90,90], [joint_length,joint_length], max_iter=1000)
    joint_theta1, joint_theta2 = angle
    joint_theta1_array = np.linspace(90, joint_theta1, 30)
    joint_theta2_array = np.linspace(90, joint_theta2, 30)
    
    #touches the point
    path_k = []
    
    for i in range(0, 30):
        path_k.append([joint_theta1_array[i], joint_theta2_array[i], theta1])
    
    for i in range(0, 30):
        path.append(path_k[i])
    #     #print("path_k",path_k)
    
    #returns from the point
    for i in range(29, -1, -1):
        path.append(path_k[i])

    # #rotates to the next point
    if theta1 > theta2:
        for i in range(theta1, theta2, -1):
            path.append([90, 90, i])
    else:
        for i in range(theta1,theta2):
            path.append([90, 90, i])
        
    distance2 = math.sqrt(positions[1][0]**2 + positions[1][1]**2) 
    angle, err, solved, iteration = IK([-distance2,0,0], [90,90], [joint_length,joint_length], max_iter=1000)
    joint_theta3, joint_theta4 = angle
    joint_theta3_array = np.linspace(90, joint_theta3, 30)
    joint_theta4_array = np.linspace(90, joint_theta4, 30)
    
    #touches the point
    path_k2 = []
    for i in range(0, 30):
        path_k2.append([joint_theta3_array[i], joint_theta4_array[i], theta2])
    
    for i in range(0, 30):
        path.append(path_k2[i])
    
    #returns from the point
    for i in range(29, -1, -1):
        path.append(path_k2[i])
        
        
    #rotates to the home position
    if theta2 > 180:
        for i in range(theta2, 180, -1):
            path.append([90, 90, i])
    else: 
        for i in range(theta2, 180):
            path.append([90, 90, i])

    # print(path)
    return path

def get_path2():
    joint_length = 100

    angle, err, solved, iteration = IK([-200,200,0], [90,90], [joint_length,joint_length], max_iter=1000)
    joint_theta1, joint_theta2 = angle
    joint_theta1_array = np.linspace(90, joint_theta1, 30)
    joint_theta2_array = np.linspace(90, joint_theta2, 30)
    
    #touches the point
    path_k = []
    path = []
    
    for i in range(0, 30):
        path.append([joint_theta1_array[i], joint_theta2_array[i], 180])
        path_k.append([joint_theta1_array[i], joint_theta2_array[i], 180])
    
    # for i in range(29, -1, -1):
    #     path.append(path_k[i])
        

    angle2, err, solved, iteration = IK([-100,0,0], [90,90], [joint_length,joint_length], max_iter=1000)
    joint_theta3, joint_theta4 = angle2
    joint_theta3_array = np.linspace(joint_theta1, joint_theta3, 30)
    joint_theta4_array = np.linspace(joint_theta2, joint_theta4, 30)

    for i in range(0, 30):
        path.append([joint_theta3_array[i], joint_theta4_array[i], 180])
        path_k.append([joint_theta3_array[i], joint_theta4_array[i], 180])

    for i in range(29, -1, -1):
        path.append(path_k[i])


    return path

