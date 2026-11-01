import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================
# 默认起点终点
# =========================
try:
    from config import START, GOAL
except ImportError:
    START = (1.0, 1.0)
    GOAL = (19.0, 19.0)

MAP_MIN = 0.0
MAP_MAX = 20.0
GRID_RESOLUTION = 0.5


# =========================
# 地图生成
# =========================
class MapEnv:
    def generate_mixed(self):
        np.random.seed(42)

        boundary_obs = [
            (0, 10, 0.85), (20, 10, 0.85),
            (10, 0, 0.85), (10, 20, 0.85)
        ]

        n_clusters = np.random.randint(5, 9)
        cluster_centers = []
        for _ in range(n_clusters):
            cx = np.random.uniform(3, 17)
            cy = np.random.uniform(3, 17)
            cluster_centers.append((cx, cy))

        obs = []

        for cx, cy in cluster_centers:
            n_per_cluster = np.random.randint(5, 9)
            for _ in range(n_per_cluster):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 2.0)
                x = cx + dist * np.cos(angle)
                y = cy + dist * np.sin(angle)
                r = np.random.uniform(0.3, 0.7)
                obs.append((x, y, r))

        n_background = np.random.randint(20, 35)
        for _ in range(n_background):
            x = np.random.uniform(1.5, 18.5)
            y = np.random.uniform(1.5, 18.5)
            r = np.random.uniform(0.3, 0.7)
            obs.append((x, y, r))

        hotspot_center = (np.random.uniform(8, 12), np.random.uniform(8, 12))
        n_hotspot = np.random.randint(8, 15)
        for _ in range(n_hotspot):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, 1.8)
            x = hotspot_center[0] + dist * np.cos(angle)
            y = hotspot_center[1] + dist * np.sin(angle)
            r = np.random.uniform(0.25, 0.5)
            obs.append((x, y, r))

        final_obs = boundary_obs.copy()

        def is_overlap(x, y, r, existing_list):
            for ox, oy, orr in existing_list:
                if np.hypot(x - ox, y - oy) < (r + orr + 0.12):
                    return True
            return False

        for x, y, r in obs:
            if not is_overlap(x, y, r, final_obs):
                final_obs.append((x, y, r))

        return final_obs


# =========================
# 几何工具
# =========================
def path_length(path):
    if len(path) < 2:
        return 0.0
    return float(sum(
        np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ))


def min_obstacle_dist(p, obstacles):
    x, y = p
    dmin = float("inf")
    for ox, oy, r in obstacles:
        d = np.hypot(x - ox, y - oy) - r
        if d < dmin:
            dmin = d
    return dmin


def collision(p, obstacles):
    return min_obstacle_dist(p, obstacles) <= 0.0


def segment_collision(p1, p2, obstacles):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    vv = np.dot(v, v)

    for (ox, oy, r) in obstacles:
        o = np.array([ox, oy], dtype=float)
        if vv == 0:
            dist = np.linalg.norm(o - p1)
        else:
            w = o - p1
            t = np.dot(w, v) / vv
            t = max(0.0, min(1.0, t))
            closest = p1 + t * v
            dist = np.linalg.norm(o - closest)
        if dist <= r:
            return True
    return False


# =========================
# A* 搜索
# =========================
def point_in_bounds(p):
    return MAP_MIN <= p[0] <= MAP_MAX and MAP_MIN <= p[1] <= MAP_MAX


def node_to_world(ix, iy, res=GRID_RESOLUTION):
    return (MAP_MIN + ix * res, MAP_MIN + iy * res)


def world_to_node(p, res=GRID_RESOLUTION):
    return (int(round((p[0] - MAP_MIN) / res)), int(round((p[1] - MAP_MIN) / res)))


def nearest_free_node(start_xy, obstacles, res=GRID_RESOLUTION, max_expand=30):
    sx, sy = world_to_node(start_xy, res)
    for radius in range(max_expand + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                ix, iy = sx + dx, sy + dy
                wx, wy = node_to_world(ix, iy, res)
                if not point_in_bounds((wx, wy)):
                    continue
                if not collision((wx, wy), obstacles):
                    return (ix, iy)
    return (sx, sy)


def astar(start, goal, obstacles, res=GRID_RESOLUTION):
    max_idx = int(round((MAP_MAX - MAP_MIN) / res))
    start_node = nearest_free_node(start, obstacles, res=res)
    goal_node = nearest_free_node(goal, obstacles, res=res)

    def heuristic(a, b):
        ax, ay = a
        bx, by = b
        return math.hypot(ax - bx, ay - by)

    def node_free(ix, iy):
        wx, wy = node_to_world(ix, iy, res)
        return point_in_bounds((wx, wy)) and (not collision((wx, wy), obstacles))

    neighbors = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        (0, -1, 1.0),                          (0, 1, 1.0),
        (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2))
    ]

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_node))
    came_from = {}
    g_score = {start_node: 0.0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal_node:
            nodes = [current]
            while current in came_from:
                current = came_from[current]
                nodes.append(current)
            nodes.reverse()
            return [node_to_world(ix, iy, res) for ix, iy in nodes]

        cx, cy = current
        c_world = node_to_world(cx, cy, res)

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx > max_idx or ny > max_idx:
                continue
            if not node_free(nx, ny):
                continue

            n_world = node_to_world(nx, ny, res)

            if segment_collision(c_world, n_world, obstacles):
                continue

            tentative_g = g_score[current] + step_cost * res
            neighbor = (nx, ny)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal_node) * res
                heapq.heappush(open_heap, (f, neighbor))

    return None


# =========================
# 适应度函数
# =========================
def fitness(path, obstacles,
            safe_radius=0.005,
            angle_penalty_coef=0.2,
            path_penalty=0.5,
            boundary_penalty=100.0,
            collision_penalty=40.0):

    length = path_length(path)

    safe = 0.0
    max_angle = 0.0
    for p in path:
        d = min_obstacle_dist(p, obstacles)
        if d < safe_radius:
            safe += (safe_radius - d) ** 2
    safe *= 15.0

    collide = False
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        if segment_collision(p1, p2, obstacles):
            collide = True
            break
    if not collide:
        collide = any(collision(p, obstacles) for p in path)

    angle_penalty_raw = 0.0
    if len(path) >= 3:
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i - 1])
            v2 = np.array(path[i + 1]) - np.array(path[i])

            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)

                if angle > max_angle:
                    max_angle = angle
                angle_penalty_raw += angle

        angle_penalty_raw = angle_penalty_raw - max_angle + max_angle * 2

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist_before = np.hypot(p1[0] - GOAL[0], p1[1] - GOAL[1])
            dist_after = np.hypot(p2[0] - GOAL[0], p2[1] - GOAL[1])
            if dist_after > dist_before:
                angle_penalty_raw += 0.2 * (dist_after - dist_before)

    angle_penalty = angle_penalty_raw * angle_penalty_coef

    boundary = 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = 0.0, 20.0
    for p in path:
        if p[0] < x_min:
            boundary += (x_min - p[0]) ** 2
        elif p[0] > x_max:
            boundary += (p[0] - x_max) ** 2
        if p[1] < y_min:
            boundary += (y_min - p[1]) ** 2
        elif p[1] > y_max:
            boundary += (p[1] - y_max) ** 2
    boundary *= boundary_penalty

    total = path_penalty * length + 0.05 * safe + angle_penalty + boundary
    if collide:
        total += collision_penalty

    return total, {
        "length": length,
        "safe": safe,
        "angle_penalty_raw": angle_penalty_raw,
        "angle_penalty": angle_penalty,
        "boundary": boundary,
        "collide": collide,
        "total": total
    }


# =========================
# 可视化
# =========================
def plot_scene(obstacles, path, title="A* Path"):
    fig, ax = plt.subplots(figsize=(9, 9))

    for ox, oy, r in obstacles:
        circ = Circle((ox, oy), r, color="gray", alpha=0.45)
        ax.add_patch(circ)

    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-o", linewidth=2.5, markersize=5, label="Path")

        for i, (x, y) in enumerate(path):
            ax.text(x + 0.08, y + 0.08, str(i + 1), fontsize=9)

    ax.scatter([START[0]], [START[1]], c="green", s=120, marker="o", label="Start")
    ax.scatter([GOAL[0]], [GOAL[1]], c="red", s=120, marker="*", label="Goal")

    ax.set_xlim(MAP_MIN, MAP_MAX)
    ax.set_ylim(MAP_MIN, MAP_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.show()


# =========================
# 主程序
# =========================
def main():
    env = MapEnv()
    obstacles = env.generate_mixed()

    raw_path = astar(START, GOAL, obstacles, res=GRID_RESOLUTION)
    if raw_path is None:
        print("A* 没有找到路径。")
        plot_scene(obstacles, None, title="A* Failed")
        return

    total, info = fitness(raw_path, obstacles)

    angle_percent = 0.0
    if abs(info["total"]) > 1e-12:
        angle_percent = 100.0 * info["angle_penalty"] / info["total"]

    print("======== A* 路径结果 ========")
    print(f"路径点数: {len(raw_path)}")
    print("路径点:")
    for i, p in enumerate(raw_path):
        print(f"  P{i+1}: ({p[0]:.4f}, {p[1]:.4f})")

    print("\n======== 适应度分解 ========")
    print(f"路径长度 length              = {info['length']:.6f}")
    print(f"安全惩罚 safe                = {info['safe']:.6f}")
    print(f"角度惩罚原始 angle_raw       = {info['angle_penalty_raw']:.6f}")
    print(f"角度惩罚 angle_penalty      = {info['angle_penalty']:.6f}")
    print(f"边界惩罚 boundary           = {info['boundary']:.6f}")
    print(f"是否碰撞 collide            = {info['collide']}")
    print(f"总适应度 total               = {info['total']:.6f}")
    print(f"角度惩罚占总适应度百分比     = {angle_percent:.2f}%")

    plot_scene(obstacles, raw_path, title="A* Path without Resampling")

    return raw_path, obstacles, info


if __name__ == "__main__":
    main()