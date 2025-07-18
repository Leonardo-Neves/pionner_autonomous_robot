def bresenham_line(start_node, end_node):
    """
    Yields all grid cells on a straight line between two nodes using
    Bresenham's line algorithm.
    """
    x0, y0 = start_node
    x1, y1 = end_node
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def prune_path(path, grid):
    """
    Removes redundant waypoints from a path using a line-of-sight check.

    :param path: The original list of (row, col) waypoints.
    :param grid: The occupancy grid (1=obstacle, 0=free).
    :return: A new, shorter list of waypoints.
    """
    if not path:
        return []
    
    pruned_path = [path[0]]
    current_index = 0

    while current_index < len(path) - 1:
        last_visible_index = current_index + 1
        for next_index in range(current_index + 1, len(path)):
            # Check for line-of-sight between current and next waypoints
            line_is_clear = True
            line_nodes = bresenham_line(path[current_index], path[next_index])
            for node in line_nodes:
                if grid[node[0]][node[1]] == 1: # If any cell on the line is an obstacle
                    line_is_clear = False
                    break
            
            if line_is_clear:
                # We can see this far, so keep trying further
                last_visible_index = next_index
            else:
                # The line is blocked, so the previous waypoint was the last visible one
                break
        
        # Add the last visible waypoint and start the process again from there
        pruned_path.append(path[last_visible_index])
        current_index = last_visible_index

    return pruned_path