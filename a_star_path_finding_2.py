import os
import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from matplotlib.animation import FuncAnimation

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from pathfinding.core.diagonal_movement import DiagonalMovement

# Occupacy Grid Figure
plt.ion()  # Turn on interactive mode

fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
time_log, sp_log, pv_log, err_log = [], [], [], []
p_log, i_log, d_log, out_log = [], [], [], []

def update_plot(time, sp, err):
    ax1.clear()
    ax1.plot(time, sp, 'k--', label='Setpoint (Target)')
    ax1.plot(time, err, 'r-', label='Error')
    ax1.set_title(f'PID Performance {err[len(err)-1]:.2f}')
    ax1.set_ylabel('Angle')
    plt.ylim(-2, 2)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    plt.pause(0.01)


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

    occupancy_grid = np.zeros(grid_size)

    while counter < len(TARGET_ORIENTATIONS):
        now = time.time()
        dt = now - lastTime

        target_orientation = TARGET_ORIENTATIONS[counter]

        _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
        _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)

        theta_robot = robot_ori[2]
        
        angle_error = target_orientation - theta_robot
    
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

            occupancy_grid = updateOccupancyGrid(clientID, occupancy_grid, robotHandle, laser_data, grid_size, cell_size)

        wl = v/r - (w*L)/(2*r)
        wr = v/r + (w*L)/(2*r)
        
        sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)
        
        t += dt
        lastTime = now

    return occupancy_grid

def normalizeAngle(angle):
    """ Normalize angle to the range [-pi, pi] """
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def moveTargetPosition(clientID, robotHandle, occupancy_grid, target_position, grid_size, cell_size):

    # A* Path Planning
    _, pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
    start_cell = worldToOccupacyGrid(pos, grid_size, cell_size)
    goal_cell = worldToOccupacyGrid(target_position, grid_size, cell_size)

    grid_finder = np.logical_not(occupancy_grid).astype(int)

    grid = Grid(matrix=grid_finder)

    start_node = grid.node(start_cell[1], start_cell[0]) 
    end_node = grid.node(goal_cell[1], goal_cell[0])

    finder = AStarFinder()

    path_tuples, runs = finder.find_path(start_node, end_node, grid)

    path = [(p.y, p.x) for p in path_tuples]

    path_world = [occupacyGridToWorld(p, grid_size, cell_size) for p in path]

    print(grid.grid_str(path=path_tuples, start=start_node, end=end_node))
    
    # PID

    L = 0.381  # Metros
    r = 0.0975 # Metros

    v = 0.4

    maxv = 1.0
    maxw = np.deg2rad(45)

    integral_error_angle = 0.0
    previous_error_angle = 0.0

    integral_error_distance = 0.0
    previous_error_distance = 0.0

    t = 0
    startTime = time.time()
    lastTime = startTime

    counter = 0

    loop_counter = 0

    _, referenceFrame = sim.simxGetObjectHandle(clientID, 'ReferenceFrame', sim.simx_opmode_oneshot_wait)

    while counter < len(path_world):
        now = time.time()
        dt = now - lastTime

        if dt == 0:
            dt = 0.0001 

        # plt.imshow(occupancy_grid, cmap='gray_r', origin='lower')
        # plt.title("Occupancy Grid")
        # plt.show()

        target_pos = path_world[counter]

        # print(f'target_pos: {target_pos}')

        sim.simxSetObjectPosition(clientID, referenceFrame, -1, target_pos, sim.simx_opmode_oneshot_wait)

        _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
        _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)

        theta_robot = robot_ori[2]

        # Distance regarding next point from A* path

        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]


        distance_to_target = np.sqrt((dx)**2 + (dy)**2)

        angle_to_target = np.arctan2((target_pos[1] - robot_pos[1]), (target_pos[0] - robot_pos[0]))

        # angle_error = angle_to_target - theta_robot
        # angle_error = ((angle_error + np.pi) % (2 * np.pi) - np.pi) # Normalization [-pi, pi]


        # -------------------- Professor ---------------------

        alpha = normalizeAngle(-theta_robot + np.arctan2(dy, dx))
        beta = normalizeAngle(angle_to_target - np.arctan2(dy, dx))

        kr = 4 / 20
        ka = 8 / 20
        kb = -1.5 / 20

        if abs(alpha) > np.pi/2:
            kr = -kr       
            alpha = normalizeAngle(alpha-np.pi)
            beta = normalizeAngle(beta-np.pi)
        
        v = kr*distance_to_target
        w = ka*alpha + kb*beta

        v = max(min(v, maxv), -maxv)
        w = max(min(w, maxw), -maxw)
                
        # -------------------- Angle ---------------------

        # Kp_w = 0.2
        # Ki_w = 0.01
        # Kd_w = 0.008

        # P Controll

        # p_w = Kp_w * angle_error

        # I Controll

        # integral_error_angle += angle_error * dt
        # i_w = Ki_w* integral_error_angle    

        # D Controll

        # derivative_error = (angle_error - previous_error_angle) / dt
        # d_w = Kd_w * derivative_error

        # previous_error_angle = angle_error

        # w = p_w + i_w

        # -------------------- Distance ---------------------

        # Kp_v = 0.4
        # Ki_v = 0.0002
        # Kd_v = 1


        # P Controll
        # p_v = Kp_v * distance_to_target

        # I Controll

        # integral_error_distance += distance_to_target * dt
        # i_v = Ki_v* integral_error_distance    

        # D Controll

        # derivative_error = (distance_to_target - previous_error_distance) / dt
        # d_v = Kd_v * derivative_error

        # v = p_v 

        if distance_to_target < 0.25:
            print(f"{counter}/{len(path_world)}")

            counter += 1

            # integral_error_angle = 0.0
            # integral_error_distance = 0.0

            time_log = []
            err_log = []

            if counter >= len(path_world):
                v = 0
                w = 0

        # if abs(angle_error) > np.deg2rad(20):
        #     v = 0.05

        loop_counter += 1

        
        # time_log.append(t)
        # err_log.append(distance_to_target)
        # err_log.append(angle_error)
        # update_plot(time_log, [0.15] * len(err_log), err_log)

        # Isso é o modelo cinemático, estudaremos detalhadamente depois!
        # wl = v/r - (w*L)/(2*r)
        # wr = v/r + (w*L)/(2*r)

        wr = ((2.0*v) + (w*L))/(2.0*r)
        wl = ((2.0*v) - (w*L))/(2.0*r)
        
        # Enviando velocidades
        sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)
        
        t += dt
        lastTime = now

        # time.sleep(0.01)

def inflate_obstacles(grid, robot_radius, cell_size):

    inflation_radius_cells = math.ceil(robot_radius / cell_size)
    
    y, x = np.ogrid[-inflation_radius_cells:inflation_radius_cells+1, -inflation_radius_cells:inflation_radius_cells+1]
    structure = x**2 + y**2 <= inflation_radius_cells**2

    inflated_grid = binary_dilation(grid, structure=structure)
    
    return inflated_grid.astype(int)



sim.simxFinish(-1)

clientID = sim.simxStart(
    '127.0.0.1',
    19999,
    True,
    True,
    5000,
    5
)

if clientID!=-1:

    robotname = 'Pioneer_p3dx'
    laser_range_data = "hokuyo_range_data"
    laser_angle_data = "hokuyo_angle_data"

    CELL_SIZE = 0.5  # Each grid cell is 10x10 cm
    GRID_SIZE = (100, 100) # 10x10 meters grid
    

    returnCode, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)

    returnCode, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    returnCode, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)



    _, point1 = sim.simxGetObjectHandle(clientID, 'point1', sim.simx_opmode_oneshot_wait)
    _, point1_pos = sim.simxGetObjectPosition(clientID, point1, -1, sim.simx_opmode_blocking)

    _, point2 = sim.simxGetObjectHandle(clientID, 'point2', sim.simx_opmode_oneshot_wait)
    _, point2_pos = sim.simxGetObjectPosition(clientID, point2, -1, sim.simx_opmode_blocking)

    _, point3 = sim.simxGetObjectHandle(clientID, 'point3', sim.simx_opmode_oneshot_wait)
    _, point3_pos = sim.simxGetObjectPosition(clientID, point3, -1, sim.simx_opmode_blocking)

    _, point4 = sim.simxGetObjectHandle(clientID, 'point4', sim.simx_opmode_oneshot_wait)
    _, point4_pos = sim.simxGetObjectPosition(clientID, point4, -1, sim.simx_opmode_blocking)

    # TARGET_POSITIONS = [
    #     point1_pos, 
    #     point2_pos, 
    #     point3_pos, 
    #     point4_pos
    # ]

    TARGET_POSITIONS = [
        (4, -4), 
        (-3.5, -4), 
        (-3, 3.5), 
        (3.7, 3.25)
        # (3.5, 3.5)
    ]

    # Geralmente a primeira leitura é inválida (atenção ao Operation Mode)
    # Em loop até garantir que as leituras serão válidas
    returnCode = 1
    while returnCode != 0:
        returnCode, range_data = sim.simxGetStringSignal(clientID, laser_range_data, sim.simx_opmode_streaming + 10)
    
    # Occupancy grid
    occupancy_grid = createOccupancyGridAndPathFinding(GRID_SIZE, CELL_SIZE, clientID, robotHandle, laser_range_data, laser_angle_data)

    # occupancy_grid = inflate_obstacles(occupancy_grid, robot_radius=0.1, cell_size=CELL_SIZE)

    # plt.imshow(occupancy_grid, cmap='gray_r', origin='lower')
    # plt.title("Occupancy Grid")
    # plt.show()

    for position in TARGET_POSITIONS:
        # TARGET_POSITION = (position[1], position[0])
        TARGET_POSITION = position
    
        print(f'Position: {TARGET_POSITION}')

        moveTargetPosition(clientID, robotHandle, occupancy_grid, TARGET_POSITION, GRID_SIZE, CELL_SIZE)

    # Parando o robô    
    sim.simxSetJointTargetVelocity(clientID, r_wheel, 0, sim.simx_opmode_oneshot_wait)
    sim.simxSetJointTargetVelocity(clientID, l_wheel, 0, sim.simx_opmode_oneshot_wait)

    plt.ioff()
    plt.show()

    # Parando a simulação     
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)   

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
