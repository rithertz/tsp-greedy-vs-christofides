# ============================================
# FULL CHRISTOFIDES (WITH MATCHING) + VISUALIZER
# GREEDY vs CHRISTOFIDES (PYGAME SPLIT SCREEN)
# ============================================

import pygame
import random
import math
import time
from itertools import combinations

WIDTH, HEIGHT = 1200, 600
HALF = WIDTH // 2
WHITE = (255,255,255)
GREEN = (0,255,0)
RED = (255,0,0)
BLACK = (0,0,0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Full Christofides vs Greedy")

# ---------------- UTIL ----------------
def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------------- POINTS ----------------
def generate_points(n):
    return [(random.randint(50, HALF-50), random.randint(50, HEIGHT-50)) for _ in range(n)]

# ---------------- GREEDY ----------------
def greedy(points):
    n = len(points)
    visited = [False]*n
    tour = [0]
    visited[0] = True

    for _ in range(n-1):
        last = tour[-1]
        nxt = min((dist(points[last], points[i]), i) for i in range(n) if not visited[i])[1]
        tour.append(nxt)
        visited[nxt] = True

    tour.append(0)
    return tour

# ---------------- MST (PRIM) ----------------
def prim(points):
    n = len(points)
    key = [float('inf')]*n
    parent = [-1]*n
    in_mst = [False]*n

    key[0] = 0

    for _ in range(n):
        u = min((k,i) for i,k in enumerate(key) if not in_mst[i])[1]
        in_mst[u] = True

        for v in range(n):
            d = dist(points[u], points[v])
            if not in_mst[v] and d < key[v]:
                key[v] = d
                parent[v] = u

    adj = [[] for _ in range(n)]
    for i in range(1,n):
        adj[i].append(parent[i])
        adj[parent[i]].append(i)

    return adj

# ---------------- ODD VERTICES ----------------
def get_odd_vertices(adj):
    return [i for i in range(len(adj)) if len(adj[i]) % 2 == 1]

# ---------------- PERFECT MATCHING (DP if small) ----------------
def minimum_matching(points, odd):
    n = len(odd)

    # if small → exact DP
    if n <= 16:
        memo = {}

        def dp(mask):
            if mask == 0:
                return 0, []

            if mask in memo:
                return memo[mask]

            first = (mask & -mask).bit_length() - 1
            best_cost = float('inf')
            best_pairs = []

            for j in range(first+1, n):
                if mask & (1<<j):
                    new_mask = mask ^ (1<<first) ^ (1<<j)
                    cost, pairs = dp(new_mask)
                    d = dist(points[odd[first]], points[odd[j]])
                    if cost + d < best_cost:
                        best_cost = cost + d
                        best_pairs = pairs + [(odd[first], odd[j])]

            memo[mask] = (best_cost, best_pairs)
            return memo[mask]

        return dp((1<<n)-1)[1]

    # fallback greedy matching
    unused = set(odd)
    pairs = []

    while unused:
        u = unused.pop()
        v = min(unused, key=lambda x: dist(points[u], points[x]))
        unused.remove(v)
        pairs.append((u,v))

    return pairs

# ---------------- EULER TOUR ----------------
def eulerian_tour(adj):
    graph = {i:list(adj[i]) for i in range(len(adj))}
    stack = [0]
    path = []

    while stack:
        v = stack[-1]
        if graph[v]:
            u = graph[v].pop()
            graph[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())

    return path[::-1]

# ---------------- CHRISTOFIDES ----------------
def christofides(points):
    adj = prim(points)
    odd = get_odd_vertices(adj)
    pairs = minimum_matching(points, odd)

    # add matching edges
    for u,v in pairs:
        adj[u].append(v)
        adj[v].append(u)

    euler = eulerian_tour(adj)

    # shortcut
    visited = set()
    tour = []

    for v in euler:
        if v not in visited:
            visited.add(v)
            tour.append(v)

    tour.append(tour[0])
    return tour

# ---------------- DRAW ----------------
def draw_half(points, tour, step, offset_x, color):
    for p in points:
        pygame.draw.circle(screen, WHITE, (p[0]+offset_x, p[1]), 5)

    total = 0
    for i in range(min(step, len(tour)-1)):
        a = points[tour[i]]
        b = points[tour[i+1]]
        pygame.draw.line(screen, color, (a[0]+offset_x, a[1]), (b[0]+offset_x, b[1]), 2)
        total += dist(a,b)

    return total

# ---------------- MAIN ----------------
def main():
    n = 30
    points = generate_points(n)

    start = time.time()
    g_tour = greedy(points)
    g_time = (time.time()-start)*1000

    start = time.time()
    c_tour = christofides(points)
    c_time = (time.time()-start)*1000

    step = 1
    running = True

    font = pygame.font.SysFont(None, 28)

    while running:
        pygame.time.delay(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    points = generate_points(n)
                    g_tour = greedy(points)
                    c_tour = christofides(points)
                    step = 1

        screen.fill(BLACK)

        g_dist = draw_half(points, g_tour, step, 0, GREEN)
        c_dist = draw_half(points, c_tour, step, HALF, RED)

        pygame.draw.line(screen, WHITE, (HALF,0), (HALF,HEIGHT), 2)

        t1 = font.render(f"Greedy | Dist: {int(g_dist)} | Time: {g_time:.2f} ms", True, WHITE)
        t2 = font.render(f"Christofides | Dist: {int(c_dist)} | Time: {c_time:.2f} ms", True, WHITE)

        screen.blit(t1, (20,20))
        screen.blit(t2, (HALF+20,20))

        pygame.display.update()

        step = min(step+1, len(g_tour))

    print("\n==== FINAL TELEMETRY ====")
    print("Greedy Distance:", g_dist)
    print("Christofides Distance:", c_dist)
    print("Greedy Time:", g_time, "ms")
    print("Christofides Time:", c_time, "ms")

    pygame.quit()

main()
