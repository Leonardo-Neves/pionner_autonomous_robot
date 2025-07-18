
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import numpy as np
import matplotlib.pyplot as plt
import math
import time


def readSensorData(clientId=-1, 
                    range_data_signal_id="hokuyo_range_data", 
                    angle_data_signal_id="hokuyo_angle_data"):

    # the first call should be non-blocking to avoid getting out-of-sync angle data
    returnCodeRanges, string_range_data = sim.simxGetStringSignal(clientId, range_data_signal_id, sim.simx_opmode_streaming)

    # the second call should block to avoid out-of-sync scenarios
    # between your python script and the simulator's main loop
    # (your script may be slower than the simulator's main loop, thus
    # slowing down data processing)
    returnCodeAngles, string_angle_data = sim.simxGetStringSignal(clientId, angle_data_signal_id, sim.simx_opmode_blocking)

    # check the if both data were obtained correctly
    if returnCodeRanges == 0 and returnCodeAngles == 0:
        # unpack data from range and sensor messages
        raw_range_data = sim.simxUnpackFloats(string_range_data)
        raw_angle_data = sim.simxUnpackFloats(string_angle_data)

        return raw_range_data, raw_angle_data

    # return none in case were nothing was gotten from the simulator
    return None


def create_occupancy_grid(clientId, robotHandle, laser_data, grid_size, cell_size):
    """
    Creates an occupancy grid from laser data.
    
    Returns: A 2D NumPy array representing the map. 1 for obstacle, 0 for free space.
    """
    grid = np.zeros(grid_size)
    
    # Get robot's current position and orientation
    _, robot_pos = sim.simxGetObjectPosition(clientId, robotHandle, -1, sim.simx_opmode_blocking)
    _, robot_ori = sim.simxGetObjectOrientation(clientId, robotHandle, -1, sim.simx_opmode_blocking)
    robot_theta = robot_ori[2] # Z-axis orientation (yaw)

    # World center offset for grid mapping
    world_offset_x = grid_size[0] * cell_size / 2
    world_offset_y = grid_size[1] * cell_size / 2

    for angle, distance in laser_data:
        # Don't map points that are too far away (sensor noise or max range)
        if distance > 4.9: 
            continue

        # Convert polar (laser) coordinates to Cartesian (robot-centric)
        x_local = distance * np.cos(angle)
        y_local = distance * np.sin(angle)
        
        # Rotate points to align with the world frame
        x_world_relative = x_local * np.cos(robot_theta) - y_local * np.sin(robot_theta)
        y_world_relative = x_local * np.sin(robot_theta) + y_local * np.cos(robot_theta)

        # Translate points to the robot's world position
        x_world = x_world_relative + robot_pos[0]
        y_world = y_world_relative + robot_pos[1]
        
        # Convert world coordinates to grid indices
        grid_x = int((x_world + world_offset_x) / cell_size)
        grid_y = int((y_world + world_offset_y) / cell_size)
        
        # Mark the cell as occupied
        if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
            grid[grid_y, grid_x] = 1 # Use [row, col] which corresponds to [y, x]
            
    return grid

def worldToGrid(world_coords, grid_size, cell_size):
    """Converts world coordinates (x, y) to grid cell indices (row, col)."""
    world_offset_x = grid_size[0] * cell_size / 2
    world_offset_y = grid_size[1] * cell_size / 2
    grid_x = int((world_coords[0] + world_offset_x) / cell_size)
    grid_y = int((world_coords[1] + world_offset_y) / cell_size)
    return (grid_y, grid_x) # Return as (row, col)

def gridToWorld(grid_coords, grid_size, cell_size):
    """Converts grid cell indices (row, col) to world coordinates (x, y)."""
    world_offset_x = grid_size[0] * cell_size / 2
    world_offset_y = grid_size[1] * cell_size / 2
    world_x = grid_coords[1] * cell_size - world_offset_x + cell_size / 2
    world_y = grid_coords[0] * cell_size - world_offset_y + cell_size / 2
    return (world_x, world_y)

class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # Tuple (row, col)

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

def a_star_search(grid, start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given grid.
    :param grid: 2D numpy array of the map.
    :param start: Tuple (row, col) for the start position.
    :param end: Tuple (row, col) for the end position.
    """

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node (the one with the lowest f value)
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        # Adjacent squares (8-way movement)
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            # Make sure within range
            if (node_position[0] > (len(grid) - 1) or node_position[0] < 0 or
                node_position[1] > (len(grid[0]) - 1) or node_position[1] < 0):
                continue

            # Make sure walkable terrain
            if grid[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                      ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if any(open_node for open_node in open_list if child == open_node and child.g > open_node.g):
                continue
                
            # Add the child to the open list
            open_list.append(child)
            
    return None # Path not found

# ... (after connecting to CoppeliaSim and getting handles) ...

# --- A* Integration Starts Here ---

# 1. Define Grid and Goal Parameters
CELL_SIZE = 0.1  # Each grid cell is 10x10 cm
GRID_SIZE = (100, 100) # 10x10 meters grid
GOAL_POSITION_WORLD = (0.0, 0.0) # Define your goal in world coordinates (x, y)


sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim

if clientID!=-1:

    robotname = 'Pioneer_p3dx'
    laser_range_data = "hokuyo_range_data"
    laser_angle_data = "hokuyo_angle_data"

    returnCode, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)

    returnCode, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    returnCode, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)    

    # Geralmente a primeira leitura é inválida (atenção ao Operation Mode)
    # Em loop até garantir que as leituras serão válidas
    returnCode = 1
    while returnCode != 0:
        returnCode, range_data = sim.simxGetStringSignal(clientID, laser_range_data, sim.simx_opmode_streaming + 10)
    

    # 2. Build the Occupancy Grid (once, for a static environment)
    print("Building map...")
    raw_range_data, raw_angle_data = readSensorData(clientID, laser_range_data, laser_angle_data)
    laser_data = np.array([raw_angle_data, raw_range_data]).T
    occupancy_grid = create_occupancy_grid(clientID, robotHandle, laser_data, GRID_SIZE, CELL_SIZE)
    print("Map built.")

    plt.imshow(occupancy_grid, cmap='gray_r', origin='lower')
    plt.title("Occupancy Grid")
    plt.show()

    # 3. Plan the Path using A*
    returnCode, robot_pos_world = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
    start_cell = worldToGrid(robot_pos_world, GRID_SIZE, CELL_SIZE)
    goal_cell = worldToGrid(GOAL_POSITION_WORLD, GRID_SIZE, CELL_SIZE)

    print(f"Planning path from {start_cell} to {goal_cell}...")
    path = a_star_search(occupancy_grid, start_cell, goal_cell)

    if not path:
        sim.simxFinish(clientID)
        
    print("Path found! Following...")

    # Convert path from grid cells to world coordinates
    path_world = [gridToWorld(p, GRID_SIZE, CELL_SIZE) for p in path]

    # 4. Path Following Logic
    L = 0.381  # Metros
    r = 0.0975 # Metros

    current_waypoint_index = 0
    t = 0
    startTime = time.time()
    lastTime = startTime

    while current_waypoint_index < len(path_world):
        now = time.time()
        dt = now - lastTime
        
        target_pos = path_world[current_waypoint_index]
        
        _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
        _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)

        theta = robot_ori[2]
        
        # Euclidian distance
        distance_to_target = np.sqrt((target_pos[0] - robot_pos[0])**2 + (target_pos[1] - robot_pos[1])**2)
        angle_to_target = np.arctan2((target_pos[1] - robot_pos[1]), (target_pos[0] - robot_pos[0]))
        
        # Calculate angle error
        angle_error = angle_to_target - theta
        # Normalize angle error to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        
        # Proportional Controller for path following
        v = 0.4  # Constant forward velocity
        Kp = 1.0   # Proportional gain for turning
        w = Kp * angle_error

        # If we are close to the waypoint, switch to the next one
        if distance_to_target < 0.15: # 15 cm tolerance
            current_waypoint_index += 1
            print(f"Reached waypoint. Moving to waypoint {current_waypoint_index}/{len(path_world)}")
            if current_waypoint_index >= len(path_world):
                v = 0
                w = 0
        
        # If the angle error is too large, prioritize turning over moving forward
        if abs(angle_error) > np.deg2rad(20):
            v = 0.05

        # Isso é o modelo cinemático, estudaremos detalhadamente depois!
        wl = v/r - (w*L)/(2*r)
        wr = v/r + (w*L)/(2*r)
        
        # Enviando velocidades
        sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)
        
        t += dt
        lastTime = now

    # Parando o robô    
    sim.simxSetJointTargetVelocity(clientID, r_wheel, 0, sim.simx_opmode_oneshot_wait)
    sim.simxSetJointTargetVelocity(clientID, l_wheel, 0, sim.simx_opmode_oneshot_wait)

    # Parando a simulação     
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)   

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
