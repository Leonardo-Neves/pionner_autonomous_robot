import os
import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

CELL_SIZE = 0.1  # 10x10 cm
GRID_SIZE = (100, 100) # 10x10 meters grid
TARGET_POSITION = (2.0, 2.0)

robotname = 'Pioneer_p3dx'
laser_range_data = "hokuyo_range_data"
laser_angle_data = "hokuyo_angle_data"

sim.simxFinish(-1)

clientID = sim.simxStart(
    '127.0.0.1',
    19999,
    True,
    True,
    5000,
    5
)

def readSensorData(
    clientId=-1, 
    range_data_signal_id="hokuyo_range_data", 
    angle_data_signal_id="hokuyo_angle_data"
):
    returnCodeRanges, string_range_data = sim.simxGetStringSignal(clientId, range_data_signal_id, sim.simx_opmode_streaming)
    returnCodeAngles, string_angle_data = sim.simxGetStringSignal(clientId, angle_data_signal_id, sim.simx_opmode_blocking)

    if returnCodeRanges == 0 and returnCodeAngles == 0:
        raw_range_data = sim.simxUnpackFloats(string_range_data)
        raw_angle_data = sim.simxUnpackFloats(string_angle_data)
        return raw_range_data, raw_angle_data
    return None

def worldToOccupacyGrid(world_coords, grid_size, cell_size):
    world_offset_x = grid_size[0] * cell_size / 2
    world_offset_y = grid_size[1] * cell_size / 2
    grid_x = int((world_coords[0] + world_offset_x) / cell_size)
    grid_y = int((world_coords[1] + world_offset_y) / cell_size)
    return (grid_y, grid_x)

def occupacyGridToWorld(grid_coords, grid_size, cell_size):
    world_offset_x = grid_size[0] * cell_size / 2
    world_offset_y = grid_size[1] * cell_size / 2
    world_x = grid_coords[1] * cell_size - world_offset_x + cell_size / 2
    world_y = grid_coords[0] * cell_size - world_offset_y + cell_size / 2
    return (world_x, world_y)

def updateOccupancyGrid(clientId, grid, robotHandle, laser_data, grid_size, cell_size):
    
    _, robot_pos = sim.simxGetObjectPosition(clientId, robotHandle, -1, sim.simx_opmode_blocking)
    _, robot_ori = sim.simxGetObjectOrientation(clientId, robotHandle, -1, sim.simx_opmode_blocking)
    robot_theta = robot_ori[2] # yaw in radians

    # World center offset for grid mapping
    # world_offset_x = grid_size[0] * cell_size / 2
    # world_offset_y = grid_size[1] * cell_size / 2

    for angle, distance in laser_data:
        if distance > 4.9: 
            continue

        # Polar to rectangular coordinates
        x_local = distance * np.cos(angle)
        y_local = distance * np.sin(angle)
        
        # Rotate points to align with the world frame
        x_world_relative = x_local * np.cos(robot_theta) - y_local * np.sin(robot_theta)
        y_world_relative = x_local * np.sin(robot_theta) + y_local * np.cos(robot_theta)

        # Translate points to the robot's world position
        x_world = x_world_relative + robot_pos[0]
        y_world = y_world_relative + robot_pos[1]
        
        # world to grid
        # grid_x = int((x_world + world_offset_x) / cell_size)
        # grid_y = int((y_world + world_offset_y) / cell_size)

        result = worldToOccupacyGrid((x_world, y_world), grid_size, cell_size)
        grid_y, grid_x = result
        
        if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
            grid[grid_y, grid_x] = 1 # [row, col] -> [y, x]
            
    return grid

def createOccupancyGridAndPathFinding(grid_size, cell_size, clientID, robotHandle, laser_range_data, laser_angle_data):

    L = 0.381  # Metros
    r = 0.0975 # Metros

    t = 0
    startTime = time.time()
    lastTime = startTime

    counter = 0

    TARGET_ORIENTATIONS = [np.deg2rad(90), np.deg2rad(270)]

    occupancy_grid = np.zeros(GRID_SIZE)

    while counter < len(TARGET_ORIENTATIONS):
        now = time.time()
        dt = now - lastTime

        target_orientation = TARGET_ORIENTATIONS[counter]

        _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
        _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)

        theta = robot_ori[2]
        
        angle_error = target_orientation - theta
    
        angle_error = abs((angle_error + np.pi) % (2 * np.pi) - np.pi) # Normalize[-pi, pi]
        
        v = 0.05
        Kp = 1.0
        w = Kp * angle_error

        if angle_error < abs((np.deg2rad(2) + np.pi) % (2 * np.pi) - np.pi): # 2 deg
            counter += 1

            v = 0
            w = 0

            raw_range_data, raw_angle_data = readSensorData(clientID, laser_range_data, laser_angle_data)
            laser_data = np.array([raw_angle_data, raw_range_data]).T

            occupancy_grid = updateOccupancyGrid(clientID, occupancy_grid, robotHandle, laser_data, GRID_SIZE, CELL_SIZE)

        wl = v/r - (w*L)/(2*r)
        wr = v/r + (w*L)/(2*r)
        
        sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)
        
        t += dt
        lastTime = now

if clientID != -1:

    returnCode, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)

    returnCode, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    returnCode, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)    

    returnCode = 1
    while returnCode != 0:
        returnCode, range_data = sim.simxGetStringSignal(clientID, laser_range_data, sim.simx_opmode_streaming + 10)

    L = 0.381  # Metros
    r = 0.0975 # Metros

    t = 0
    startTime = time.time()
    lastTime = startTime

    counter = 0

    TARGET_ORIENTATIONS = [np.deg2rad(90), np.deg2rad(270)]

    occupancy_grid = np.zeros(GRID_SIZE)

    while counter < len(TARGET_ORIENTATIONS):
        now = time.time()
        dt = now - lastTime

        target_orientation = TARGET_ORIENTATIONS[counter]

        _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
        _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)

        theta = robot_ori[2]
        
        angle_error = target_orientation - theta
    
        angle_error = abs((angle_error + np.pi) % (2 * np.pi) - np.pi) # Normalize[-pi, pi]
        
        v = 0.05
        Kp = 1.0
        w = Kp * angle_error

        if angle_error < abs((np.deg2rad(2) + np.pi) % (2 * np.pi) - np.pi): # 2 deg
            counter += 1

            v = 0
            w = 0

            raw_range_data, raw_angle_data = readSensorData(clientID, laser_range_data, laser_angle_data)
            laser_data = np.array([raw_angle_data, raw_range_data]).T

            occupancy_grid = updateOccupancyGrid(clientID, occupancy_grid, robotHandle, laser_data, GRID_SIZE, CELL_SIZE)

        wl = v/r - (w*L)/(2*r)
        wr = v/r + (w*L)/(2*r)
        
        sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)
        
        t += dt
        lastTime = now


    plt.imshow(occupancy_grid, cmap='gray_r', origin='lower')
    plt.title("Occupancy Grid")
    plt.show()

    # Parando o robô    
    sim.simxSetJointTargetVelocity(clientID, r_wheel, 0, sim.simx_opmode_oneshot_wait)
    sim.simxSetJointTargetVelocity(clientID, l_wheel, 0, sim.simx_opmode_oneshot_wait)

    # Parando a simulação     
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)   

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
