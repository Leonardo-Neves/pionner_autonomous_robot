import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from PythonRobotics.PathPlanning.DStarLite.d_star_lite import DStarLite, Node


# Occupacy Grid Figure
plt.ion() 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8), sharex=True)
time_log, sp_log, pv_log, err_log_distance, err_log_angle = [], [], [], [], []
p_log, i_log, d_log, out_log = [], [], [], []

def update_plot(ax, time, sp, err):
    ax.clear()
    ax.plot(time, sp, 'k--', label='Setpoint (Target)')
    ax.plot(time, err, 'r-', label='Error')
    ax.set_title(f'PID Performance {err[len(err)-1]:.2f}')
    ax.set_ylabel('Angle')
    plt.ylim(-2, 2)
    ax.legend(loc='upper right')
    ax.grid(True)

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

def inflate_obstacles(grid, robot_radius, cell_size):

    inflation_radius_cells = math.ceil(robot_radius / cell_size)
    
    y, x = np.ogrid[-inflation_radius_cells:inflation_radius_cells+1, -inflation_radius_cells:inflation_radius_cells+1]
    structure = x**2 + y**2 <= inflation_radius_cells**2

    inflated_grid = binary_dilation(grid, structure=structure)
    
    return inflated_grid.astype(int)

def worldToOccupacyGrid(world_coords, grid_size, cell_size):
    
    world_offset_x = grid_size[1] * cell_size / 2
    world_offset_y = grid_size[0] * cell_size / 2

    grid_x = int((world_coords[0] + world_offset_x) / cell_size)
    grid_y = int((world_coords[1] + world_offset_y) / cell_size)

    grid_x = max(0, min(grid_x, grid_size[1] - 1))
    grid_y = max(0, min(grid_y, grid_size[0] - 1))

    return (grid_y, grid_x)

def occupacyGridToWorld(grid_coords, grid_size, cell_size):
    
    world_offset_x = grid_size[1] * cell_size / 2
    world_offset_y = grid_size[0] * cell_size / 2

    corner_x = grid_coords[0] * cell_size - world_offset_x
    corner_y = grid_coords[1] * cell_size - world_offset_y

    world_x = corner_x + cell_size / 2
    world_y = corner_y + cell_size / 2

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

def createInitialOccupancyGridAndPathFinding(grid_size, cell_size, clientID, robotHandle, laser_range_data, laser_angle_data):

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

def moveTargetPosition(clientID, robotHandle, occupancy_grid, target_position, grid_size, cell_size, laser_range_data, laser_angle_data):

    global time_log, sp_log, pv_log, err_log_angle, err_log_distance

    # A* Path Planning

    ox, oy = [], []

    for i in range(grid_size[0]):
        ox.append(i)
        oy.append(0)
        ox.append(i)
        oy.append(grid_size[1] - 1)
        
    for i in range(grid_size[1]):
        ox.append(0)
        oy.append(i)
        ox.append(grid_size[0] - 1)
        oy.append(i)

    for y in range(occupancy_grid.shape[0]):
        for x in range(occupancy_grid.shape[1]):
            if occupancy_grid[y, x] == 1:
                ox.append(y)
                oy.append(x)

    dstar = DStarLite(ox, oy)

    _, pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)

    start_cell = worldToOccupacyGrid(pos, grid_size, cell_size)
    goal_cell = worldToOccupacyGrid(target_position, grid_size, cell_size)

    dstar.initialize(
        Node(x=start_cell[0], y=start_cell[1]),
        Node(x=goal_cell[0], y=goal_cell[1])
    )

    dstar.compute_shortest_path()
    path_grid_coords = dstar.compute_current_path()

    path_realigned = []
    for node in path_grid_coords:
        realigned_x = node.x + dstar.x_min_world
        realigned_y = node.y + dstar.y_min_world
        path_realigned.append((realigned_y, realigned_x))

    
    planned_path_world_x = [node[0] for node in path_realigned]
    planned_path_world_y = [node[1] for node in path_realigned]

    ax1.clear()
    ax1.imshow(occupancy_grid, cmap='gray_r', origin='lower')
    ax1.plot(planned_path_world_x, planned_path_world_y, 'r-', label='Planned Path')
    ax1.set_title(f'Occupacy Grid')
    ax1.set_ylabel('Angle')
    
    ax1.legend(loc='upper right')
    ax1.grid(True)

    plt.pause(0.01)
    
    
    # # PID

    L = 0.381  # Metros
    r = 0.0975 # Metros

    _, referenceFrame = sim.simxGetObjectHandle(clientID, 'NextTargetPoint', sim.simx_opmode_oneshot_wait)
    
    for i, waypoint_grid in enumerate(path_realigned):

        target_pos_world = occupacyGridToWorld(waypoint_grid, grid_size, cell_size)


        sim.simxSetObjectPosition(clientID, referenceFrame, -1, (target_pos_world[0], target_pos_world[1], 0), sim.simx_opmode_oneshot_wait)
        

        print(f"Moving to waypoint {i+1}/{len(path_realigned)}: Grid {waypoint_grid} -> World ({target_pos_world[0]:.2f}, {target_pos_world[1]:.2f})")
        
        integral_error_angle = 0.0
        previous_error_angle = 0.0
        
        time_log, err_log_angle, err_log_distance = [], [], []
        t = 0
        startTime = time.time()
        lastTime = startTime
        
        distance_to_target = float('inf')

        while distance_to_target > 0.15:

            now = time.time()
            dt = now - lastTime

            if dt == 0: 
                continue

            _, robot_pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_blocking)
            _, robot_ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_blocking)
            yaw_robot = robot_ori[2]

            dx = target_pos_world[0] - robot_pos[0]
            dy = target_pos_world[1] - robot_pos[1]
            distance_to_target = np.sqrt(dx**2 + dy**2)
            angle_to_target = np.arctan2(dy, dx)
            
            angle_error = angle_to_target - yaw_robot
            angle_error = normalizeAngle(angle_error)

            Kp_w, Ki_w, Kd_w = 1.3, 0.24, 0.001

            integral_error_angle += angle_error * dt

            derivative_error = (angle_error - previous_error_angle) / dt

            w = (Kp_w * angle_error) + (Ki_w * integral_error_angle) + (Kd_w * derivative_error)

            previous_error_angle = angle_error

            v = 0.4

            if distance_to_target < 0.5:
                v = 0.2

            if i == len(path_realigned) - 1 and distance_to_target < 0.15:
                v = 0
                w = 0

            # w = -w

            wl = ((2.0 * v) - (w * L)) / (2.0 * r)
            wr = ((2.0 * v) + (w * L)) / (2.0 * r)
            
            sim.simxSetJointTargetVelocity(clientID, l_wheel, wl, sim.simx_opmode_streaming)
            sim.simxSetJointTargetVelocity(clientID, r_wheel, wr, sim.simx_opmode_streaming)

            t += dt
            lastTime = now

            time_log.append(t)
            err_log_distance.append(distance_to_target)
            err_log_angle.append(angle_error)

            update_plot(ax2, time_log, [0.15] * len(err_log_distance), err_log_distance)
            update_plot(ax3, time_log, [0] * len(err_log_angle), err_log_angle)


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

    CELL_SIZE = 0.1  # Each grid cell is 10x10 cm
    GRID_SIZE = (100, 100) # 10x10 meters grid
    

    returnCode, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)

    returnCode, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    returnCode, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)

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
    
    occupancy_grid = np.zeros(GRID_SIZE)

    occupancy_grid = createInitialOccupancyGridAndPathFinding(GRID_SIZE, CELL_SIZE, clientID, robotHandle, laser_range_data, laser_angle_data)

    occupancy_grid = inflate_obstacles(occupancy_grid, robot_radius=0.3, cell_size=CELL_SIZE)

    for position in TARGET_POSITIONS:
        
        TARGET_POSITION = position
    
        print(f'Position: {TARGET_POSITION}')

        moveTargetPosition(clientID, robotHandle, occupancy_grid, TARGET_POSITION, GRID_SIZE, CELL_SIZE, laser_range_data, laser_angle_data)

    # Parando o robô    
    sim.simxSetJointTargetVelocity(clientID, r_wheel, 0, sim.simx_opmode_oneshot_wait)
    sim.simxSetJointTargetVelocity(clientID, l_wheel, 0, sim.simx_opmode_oneshot_wait)

    plt.ioff()
    plt.show()

    # Parando a simulação     
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)   

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
