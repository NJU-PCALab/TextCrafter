import gurobipy as gp
from gurobipy import GRB
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def generate_rectangles_gurobi(points, min_area, aspect_ratio_min=1.5, aspect_ratio_max=2.5):
    """
    Generates a rectangle corresponding to a given point, using the Gurobi.

    Parameters:
    - points: List of tuples [(x1, y1), (x2, y2), ...], each tuple represents the coordinates of a point.
    - min_area: Minimum area of the rectangle.
    - aspect_ratio_min: Minimum aspect ratio of the rectangle.
    - aspect_ratio_max: Maximum aspect ratio of the rectangle.

    Returns:
    - List of dictionaries, each dictionary contains (m_offset, n_offset, m_scale, n_scale) of the rectangle.
    """
    N = len(points)
    model = gp.Model("Rectangle_Placement")

    m_offset = model.addVars(N, lb=0.0, ub=1.0, name="m_offset")
    n_offset = model.addVars(N, lb=0.0, ub=1.0, name="n_offset")
    m_scale = model.addVars(N, lb=0.0, ub=1.0, name="m_scale")
    n_scale = model.addVars(N, lb=0.0, ub=1.0, name="n_scale")

    # Add area, border and aspect ratio constraints
    for i in range(N):
        model.addConstr(m_scale[i] * n_scale[i] >= min_area, name=f"MinArea_{i}")
        model.addConstr(m_offset[i] + m_scale[i] <= 1.0, name=f"RightBoundary_{i}")
        model.addConstr(n_offset[i] + n_scale[i] <= 1.0, name=f"BottomBoundary_{i}")
        model.addConstr(m_scale[i] >= aspect_ratio_min * n_scale[i], name=f"AspectRatioMin_{i}")
        model.addConstr(m_scale[i] <= aspect_ratio_max * n_scale[i], name=f"AspectRatioMax_{i}")

    # Non-overlapping constraints
    M = 1.0
    binary_vars = {}
    for (i, j) in itertools.combinations(range(N), 2):
        binary_vars[(i, j, 'left')] = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
        binary_vars[(i, j, 'right')] = model.addVar(vtype=GRB.BINARY, name=f"right_{i}_{j}")
        binary_vars[(i, j, 'above')] = model.addVar(vtype=GRB.BINARY, name=f"above_{i}_{j}")
        binary_vars[(i, j, 'below')] = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}")

        model.addConstr(
            binary_vars[(i, j, 'left')] + binary_vars[(i, j, 'right')] +
            binary_vars[(i, j, 'above')] + binary_vars[(i, j, 'below')] >= 1,
            name=f"NonOverlap_{i}_{j}"
        )

        model.addConstr(
            m_offset[i] + m_scale[i] <= m_offset[j] + M * (1 - binary_vars[(i, j, 'left')]),
            name=f"Left_{i}_{j}"
        )
        model.addConstr(
            m_offset[j] + m_scale[j] <= m_offset[i] + M * (1 - binary_vars[(i, j, 'right')]),
            name=f"Right_{i}_{j}"
        )
        model.addConstr(
            n_offset[i] + n_scale[i] <= n_offset[j] + M * (1 - binary_vars[(i, j, 'above')]),
            name=f"Above_{i}_{j}"
        )
        model.addConstr(
            n_offset[j] + n_scale[j] <= n_offset[i] + M * (1 - binary_vars[(i, j, 'below')]),
            name=f"Below_{i}_{j}"
        )

    # Objective function: Minimize the sum of the Manhattan distances between the center of the rectangle and the corresponding points
    objective = gp.LinExpr()
    for i in range(N):
        center_x = m_offset[i] + 0.5 * m_scale[i]
        center_y = n_offset[i] + 0.5 * n_scale[i]
        point_x, point_y = points[i]
        dx = model.addVar(lb=-1.0, ub=1.0, name=f"dx_{i}")
        dy = model.addVar(lb=-1.0, ub=1.0, name=f"dy_{i}")
        abs_dx = model.addVar(lb=0.0, ub=1.0, name=f"abs_dx_{i}")
        abs_dy = model.addVar(lb=0.0, ub=1.0, name=f"abs_dy_{i}")

        model.addConstr(dx == center_x - point_x, name=f"dx_def_{i}")
        model.addConstr(dy == center_y - point_y, name=f"dy_def_{i}")

        model.addConstr(abs_dx >= dx, name=f"abs_dx1_{i}")
        model.addConstr(abs_dx >= -dx, name=f"abs_dx2_{i}")
        model.addConstr(abs_dy >= dy, name=f"abs_dy1_{i}")
        model.addConstr(abs_dy >= -dy, name=f"abs_dy2_{i}")

        objective += abs_dx + abs_dy

    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('OutputFlag', 0)

    # Search solution
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("No feasible solution was found. Please check the constraints or adjust the minimum area.")
        exit()

    # Extract results
    rectangles = []
    for i in range(N):
        rect = {
            'm_offset': max(m_offset[i].X, 0.0), # Make sure it is greater than 0
            'n_offset': max(n_offset[i].X, 0.0),
            'm_scale': max(m_scale[i].X, 0.0),
            'n_scale': max(n_scale[i].X, 0.0)
        }
        rectangles.append(rect)
    return rectangles


def visualize_rectangles(rectangles, points, filename="Rectangle Placement Visualization.png"):
    """
    Visualize rectangles and points, adjust the coordinate system so that (0,0) is at the top left corner, and save the image to a file.

    Parameters:
    - rectangles: List of dictionaries, each dictionary contains (m_offset, n_offset, m_scale, n_scale) of a rectangle.
    - points: List of tuples [(x1, y1), (x2, y2), ...], each tuple represents the coordinates of a point.
    - filename: The file name to save the image to, default is "Rectangle Placement Visualization.png".
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    unit_square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(unit_square)

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    for i, rect in enumerate(rectangles):
        m_offset = rect['m_offset']
        n_offset = rect['n_offset']
        m_scale = rect['m_scale']
        n_scale = rect['n_scale']

        color = colors[i % len(colors)]

        rectangle = patches.Rectangle((m_offset, n_offset), m_scale, n_scale, linewidth=1.5, edgecolor=color,
                                      facecolor=color, alpha=0.3)
        ax.add_patch(rectangle)

        center_x = m_offset + m_scale / 2
        center_y = n_offset + n_scale / 2
        ax.plot(center_x, center_y, marker='x', color=color, markersize=10, label=f"Rectangle {i + 1} Center")

    for i, point in enumerate(points):
        point_x, point_y = point
        ax.plot(point_x, point_y, marker='o', color='black', markersize=8, label=f"Point {i + 1}")
        ax.text(point_x + 0.01, point_y - 0.02, f"P{i + 1}", fontsize=12, color='black')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Rectangle Placement Visualization')
    ax.set_aspect('equal', adjustable='box')

    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    plt.grid(True)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # plt.show()

def generate_rectangles_random(area):
    rectangles = []

    for _ in range(area):
        # Generate m_offset and n_offset in [0, 0.9)
        m_offset = random.random() * 0.9
        n_offset = random.random() * 0.9

        # Generate m_scale in (0, 1-m_offset]
        max_m_scale = 1 - m_offset
        m_scale = random.uniform(0.1, max_m_scale)

        # Generate n_scale in (0, 1-n_offset]
        max_n_scale = 1 - n_offset
        n_scale = random.uniform(0.1, max_n_scale)

        # Create rectangle dictionary
        rectangle = {
            'm_offset': m_offset,
            'n_offset': n_offset,
            'm_scale': m_scale,
            'n_scale': n_scale
        }

        rectangles.append(rectangle)

    return rectangles


def generate_rectangles_fix(area):
    rectangles = []

    if area == 2:
        rectangles.append(
            dict({

            })
        )

    return rectangles


# Debug
if __name__ == "__main__":
    points = [[0.21875, 0.125], [0.640625, 0.234375], [0.984375, 0.40625]]

    min_area = 0.24

    rectangles = generate_rectangles_gurobi(points, min_area)

    if rectangles:
        for i, rect in enumerate(rectangles):
            print(f"Rectangle {i + 1}:")
            print(f"  m_offset (x): {rect['m_offset']:.4f}")
            print(f"  n_offset (y): {rect['n_offset']:.4f}")
            print(f"  m_scale (width): {rect['m_scale']:.4f}")
            print(f"  n_scale (height): {rect['n_scale']:.4f}")
            center_x = rect['m_offset'] + rect['m_scale'] / 2
            center_y = rect['n_offset'] + rect['n_scale'] / 2
            point_x, point_y = points[i]
            distance = abs(center_x - point_x) + abs(center_y - point_y)
            print(f"  Manhattan Distance to point: {distance:.4f}\n")

        visualize_rectangles(rectangles, points)
