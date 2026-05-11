#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

// -----------------------------
// POINT STRUCTURE
// -----------------------------
struct Point {
    double x, y;
};

// -----------------------------
// DISTANCE FUNCTION
// -----------------------------
double dist(Point a, Point b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

// -----------------------------
// GREEDY TSP (Nearest Neighbor)
// -----------------------------
pair<vector<int>, double> greedyTSP(const vector<Point>& pts) {
    int n = pts.size();
    vector<bool> visited(n, false);
    vector<int> tour;
    tour.push_back(0);
    visited[0] = true;

    double total = 0;

    for(int i = 1; i < n; i++) {
        int last = tour.back();
        double best = 1e18;
        int next = -1;

        for(int j = 0; j < n; j++) {
            if(!visited[j]) {
                double d = dist(pts[last], pts[j]);
                if(d < best) {
                    best = d;
                    next = j;
                }
            }
        }

        tour.push_back(next);
        visited[next] = true;
        total += best;
    }

    total += dist(pts[tour.back()], pts[0]);
    tour.push_back(0);

    return {tour, total};
}

// -----------------------------
// MINIMUM SPANNING TREE (Prim)
// -----------------------------
vector<vector<int>> primMST(const vector<Point>& pts) {
    int n = pts.size();
    vector<double> key(n, 1e18);
    vector<int> parent(n, -1);
    vector<bool> inMST(n, false);

    key[0] = 0;

    for(int i = 0; i < n; i++) {
        double mn = 1e18;
        int u = -1;
        for(int v = 0; v < n; v++) {
            if(!inMST[v] && key[v] < mn) {
                mn = key[v];
                u = v;
            }
        }

        inMST[u] = true;

        for(int v = 0; v < n; v++) {
            double d = dist(pts[u], pts[v]);
            if(!inMST[v] && d < key[v]) {
                key[v] = d;
                parent[v] = u;
            }
        }
    }

    vector<vector<int>> adj(n);
    for(int i = 1; i < n; i++) {
        adj[i].push_back(parent[i]);
        adj[parent[i]].push_back(i);
    }

    return adj;
}

// -----------------------------
// DFS TOUR (for Christofides approx shortcut)
// -----------------------------
void dfs(int u, vector<vector<int>>& adj, vector<bool>& visited, vector<int>& tour) {
    visited[u] = true;
    tour.push_back(u);
    for(int v : adj[u]) {
        if(!visited[v]) dfs(v, adj, visited, tour);
    }
}

pair<vector<int>, double> christofidesApprox(const vector<Point>& pts) {
    int n = pts.size();

    // Step 1: MST
    auto mst = primMST(pts);

    // Step 2: DFS traversal (shortcutting)
    vector<bool> visited(n, false);
    vector<int> tour;
    dfs(0, mst, visited, tour);

    tour.push_back(0);

    double total = 0;
    for(int i = 0; i < (int)tour.size()-1; i++) {
        total += dist(pts[tour[i]], pts[tour[i+1]]);
    }

    return {tour, total};
}

// -----------------------------
// VISUALIZATION (ASCII)
// -----------------------------
void visualize(const vector<Point>& pts, const vector<int>& tour, string name) {
    cout << "\n--- " << name << " TOUR ---\n";
    for(int i : tour) {
        cout << i << " -> ";
    }
    cout << "END\n";
}

// -----------------------------
// MAIN
// -----------------------------
int main() {
    int n = 100; 
    vector<Point> pts(n);

    srand(time(0));
    for(int i = 0; i < n; i++) {
        pts[i] = {rand()%1000, rand()%1000};
    }

    // GREEDY
    auto start1 = high_resolution_clock::now();
    auto greedy = greedyTSP(pts);
    auto end1 = high_resolution_clock::now();

    // CHRISTOFIDES (approx version)
    auto start2 = high_resolution_clock::now();
    auto christo = christofidesApprox(pts);
    auto end2 = high_resolution_clock::now();

    auto time1 = duration_cast<microseconds>(end1 - start1).count();
    auto time2 = duration_cast<microseconds>(end2 - start2).count();

    // VISUALIZATION
    visualize(pts, greedy.first, "GREEDY");
    visualize(pts, christo.first, "CHRISTOFIDES");

    // TELEMETRY
    cout << "\n===== TELEMETRY =====\n";
    cout << "Greedy Distance: " << greedy.second << "\n";
    cout << "Christofides Distance: " << christo.second << "\n";

    cout << "Greedy Time (microseconds): " << time1 << "\n";
    cout << "Christofides Time (microseconds): " << time2 << "\n";

    cout << "Efficiency Gain (distance ratio): " << (greedy.second / christo.second) << "\n";

    return 0;
}
