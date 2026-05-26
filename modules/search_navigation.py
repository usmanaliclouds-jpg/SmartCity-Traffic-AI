"""
Module 6: Search & Navigation Module
Description:
    The Search module performs route generation over the city road
    network. Intersections are modeled as nodes and roads as edges,
    with optional weights representing time, congestion, delay, or
    access penalties.

    Supported Algorithms:
        BFS  (Breadth-First Search):
            Used for unweighted route problems. Finds the path with
            the fewest hops (intersections). Optimal for unweighted.

        UCS  (Uniform Cost Search):
            Fallback for weighted problems when no heuristic is
            available. Expands nodes by cumulative path cost.
            Guaranteed to find the optimal-cost path.

        A*   (A-Star Search):
            Used for weighted problems when a heuristic is available.
            Combines actual cost g(n) with heuristic h(n) to guide
            search toward the goal efficiently.

    City Road Network (from project spec):

    Unweighted Graph (adjacency list):
        Police_HQ         - Traffic_Control_Center, River_Bridge
        Traffic_Control_Center - Police_HQ, North_Station
        North_Station     - Traffic_Control_Center, River_Bridge, Central_Junction
        River_Bridge      - Police_HQ, North_Station, Stadium
        Stadium           - River_Bridge, East_Market
        East_Market       - Stadium, Central_Junction, City_Hospital
        Central_Junction  - North_Station, East_Market, West_Terminal, South_Residential
        West_Terminal     - Central_Junction, Fire_Station, Industrial_Zone
        Airport_Road      - South_Residential, City_Hospital
        South_Residential - Central_Junction, Airport_Road
        City_Hospital     - East_Market, Airport_Road
        Industrial_Zone   - West_Terminal
        Fire_Station      - West_Terminal

    Weighted Graph (edge weights from project spec diagram):
        Traffic_Control_Center - Police_HQ          : 2
        Traffic_Control_Center - North_Station      : 4
        Traffic_Control_Center - Airport_Road       : 6
        North_Station          - River_Bridge       : 4
        North_Station          - South_Residential  : 3
        North_Station          - Central_Junction   : 3 (estimated)
        South_Residential      - Central_Junction   : 4
        South_Residential      - Stadium            : 5
        Stadium                - East_Market        : 3
        East_Market            - Central_Junction   : 3
        East_Market            - City_Hospital      : 3
        Central_Junction       - West_Terminal      : 3
        West_Terminal          - Fire_Station       : 2
        City_Hospital          - Airport_Road       : 3 (estimated)
"""

import heapq
from collections import deque


class SearchNavigationModule:
    """
    Implements BFS, UCS, and A* search algorithms over the city
    road network for route generation.
    """

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the Search & Navigation module with both the
            unweighted and weighted city road graphs.
        """
        # Unweighted adjacency list (BFS)
        self._unweighted_graph = {
            "Police_HQ"             : ["Traffic_Control_Center", "River_Bridge"],
            "Traffic_Control_Center": ["Police_HQ", "North_Station", "Airport_Road"],
            "North_Station"         : ["Traffic_Control_Center", "River_Bridge",
                                        "Central_Junction", "South_Residential"],
            "River_Bridge"          : ["Police_HQ", "North_Station", "Stadium"],
            "Stadium"               : ["River_Bridge", "East_Market", "South_Residential"],
            "East_Market"           : ["Stadium", "Central_Junction", "City_Hospital"],
            "Central_Junction"      : ["North_Station", "East_Market",
                                        "West_Terminal", "South_Residential"],
            "West_Terminal"         : ["Central_Junction", "Fire_Station", "Industrial_Zone"],
            "Airport_Road"          : ["Traffic_Control_Center", "South_Residential",
                                        "City_Hospital"],
            "South_Residential"     : ["Central_Junction", "Airport_Road",
                                        "North_Station", "Stadium"],
            "City_Hospital"         : ["East_Market", "Airport_Road"],
            "Industrial_Zone"       : ["West_Terminal"],
            "Fire_Station"          : ["West_Terminal"]
        }

        # Weighted adjacency list: {node: [(neighbor, weight), ...]}
        self._weighted_graph = {
            "Police_HQ"             : [("Traffic_Control_Center", 2), ("River_Bridge", 4)],
            "Traffic_Control_Center": [("Police_HQ", 2), ("North_Station", 4),
                                        ("Airport_Road", 6)],
            "North_Station"         : [("Traffic_Control_Center", 4), ("River_Bridge", 4),
                                        ("Central_Junction", 3), ("South_Residential", 3)],
            "River_Bridge"          : [("Police_HQ", 4), ("North_Station", 4),
                                        ("Stadium", 5)],
            "Stadium"               : [("River_Bridge", 5), ("East_Market", 3),
                                        ("South_Residential", 5)],
            "East_Market"           : [("Stadium", 3), ("Central_Junction", 3),
                                        ("City_Hospital", 3)],
            "Central_Junction"      : [("North_Station", 3), ("East_Market", 3),
                                        ("West_Terminal", 3), ("South_Residential", 4)],
            "West_Terminal"         : [("Central_Junction", 3), ("Fire_Station", 2),
                                        ("Industrial_Zone", 4)],
            "Airport_Road"          : [("Traffic_Control_Center", 6),
                                        ("South_Residential", 5), ("City_Hospital", 3)],
            "South_Residential"     : [("Central_Junction", 4), ("Airport_Road", 5),
                                        ("North_Station", 3), ("Stadium", 5)],
            "City_Hospital"         : [("East_Market", 3), ("Airport_Road", 3)],
            "Industrial_Zone"       : [("West_Terminal", 4)],
            "Fire_Station"          : [("West_Terminal", 2)]
        }

        # Heuristic table: approximate distance to City_Hospital (for A*)
        # Used as the default goal heuristic; updated dynamically when goal changes
        self._heuristic_to_hospital = {
            "Police_HQ"             : 10,
            "Traffic_Control_Center": 9,
            "North_Station"         : 7,
            "River_Bridge"          : 9,
            "Stadium"               : 6,
            "East_Market"           : 3,
            "Central_Junction"      : 5,
            "West_Terminal"         : 7,
            "Airport_Road"          : 3,
            "South_Residential"     : 5,
            "City_Hospital"         : 0,
            "Industrial_Zone"       : 11,
            "Fire_Station"          : 9
        }

    def find_route(self, processed_request, csp_result=None):
        """
        Function: find_route
        Description:
            Selects and executes the appropriate search algorithm
            based on request type and context, then returns the result.

            Algorithm selection:
              - Emergency request + weighted graph -> A* (fastest weighted)
              - Normal request + weighted context  -> UCS (optimal weighted)
              - Simple route request               -> BFS (fewest hops)

        Parameters:
            processed_request (dict): Validated request with source/dest.
            csp_result        (dict): CSP output; if a corridor is active,
                                      A* is preferred.

        Returns:
            dict: {
                'found'      : bool,
                'algorithm'  : str,
                'path'       : list[str],
                'total_cost' : float or None (for BFS),
                'hops'       : int,
                'reason'     : str (if not found)
            }
        """
        source      = processed_request.get("current_location", "")
        destination = processed_request.get("destination", "")
        req_type    = processed_request.get("request_category", "")
        is_emergency= processed_request.get("is_emergency_vehicle", False)
        density     = processed_request.get("traffic_density", "Low")

        # Validate nodes exist in graph
        if source not in self._unweighted_graph:
            return {
                "found"      : False,
                "algorithm"  : "None",
                "path"       : [],
                "total_cost" : None,
                "hops"       : 0,
                "reason"     : f"Source '{source}' not found in city graph."
            }
        if destination not in self._unweighted_graph:
            return {
                "found"      : False,
                "algorithm"  : "None",
                "path"       : [],
                "total_cost" : None,
                "hops"       : 0,
                "reason"     : f"Destination '{destination}' not found in city graph."
            }

        # Trivial case: already at destination
        if source == destination:
            return {
                "found"      : True,
                "algorithm"  : "Trivial",
                "path"       : [source],
                "total_cost" : 0,
                "hops"       : 0,
                "reason"     : "Source equals destination."
            }

        # Select algorithm based on context
        corridor_active = csp_result.get("corridor_active", False) if csp_result else False

        if is_emergency or corridor_active:
            # Emergency: use A* for speed with heuristic guidance
            algorithm   = "A*"
            path, cost  = self._astar(source, destination)
        elif density == "High" or req_type in (
            "Control_Allocation_Request",
            "Integrated_City_Service_Request"
        ):
            # Weighted scenario: use UCS for optimal cost
            algorithm   = "UCS"
            path, cost  = self._ucs(source, destination)
        else:
            # Standard route: BFS (fewest intersections)
            algorithm   = "BFS"
            path        = self._bfs(source, destination)
            cost        = None

        if path is None or len(path) == 0:
            return {
                "found"      : False,
                "algorithm"  : algorithm,
                "path"       : [],
                "total_cost" : None,
                "hops"       : 0,
                "reason"     : f"No route found from '{source}' to '{destination}'."
            }

        return {
            "found"      : True,
            "algorithm"  : algorithm,
            "path"       : path,
            "total_cost" : round(cost, 2) if cost is not None else None,
            "hops"       : len(path) - 1,
            "reason"     : ""
        }

    # BFS Implementation
    
    def _bfs(self, source, destination):
        """
        Function: _bfs
        Description:
            Breadth-First Search on the unweighted city graph.
            Finds the path with the fewest number of edges (hops).

        Parameters:
            source      (str): Starting node.
            destination (str): Target node.

        Returns:
            list[str] or None: Ordered path from source to destination,
                                or None if no path exists.
        """
        # Queue holds: (current_node, path_so_far)
        queue   = deque()
        queue.append((source, [source]))
        visited = set()
        visited.add(source)

        while queue:
            current_node, path = queue.popleft()

            if current_node == destination:
                return path

            neighbors = self._unweighted_graph.get(current_node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    # UCS Implementation
    
    def _ucs(self, source, destination):
        """
        ---------------------------------------------------------------
        Function: _ucs
        Description:
            Uniform Cost Search on the weighted city graph.
            Guarantees optimal (lowest-cost) path.

        Parameters:
            source      (str): Starting node.
            destination (str): Target node.

        Returns:
            tuple: (path: list[str], total_cost: float)
                   or (None, None) if no path exists.
        ---------------------------------------------------------------
        """
        # Priority queue: (cumulative_cost, node, path)
        priority_queue = []
        heapq.heappush(priority_queue, (0, source, [source]))

        visited = {}

        while priority_queue:
            current_cost, current_node, path = heapq.heappop(priority_queue)

            # Skip if we've already found a cheaper path to this node
            if current_node in visited and visited[current_node] <= current_cost:
                continue
            visited[current_node] = current_cost

            if current_node == destination:
                return path, current_cost

            neighbors = self._weighted_graph.get(current_node, [])
            for neighbor, edge_weight in neighbors:
                new_cost = current_cost + edge_weight
                if neighbor not in visited or visited.get(neighbor, float("inf")) > new_cost:
                    heapq.heappush(priority_queue, (new_cost, neighbor, path + [neighbor]))

        return None, None  # No path found

    # A* Implementation
   
    def _heuristic(self, node, goal):
        """
        Function: _heuristic
        Description:
            Heuristic function for A*. Returns an estimated cost from
            'node' to 'goal'. Uses precomputed values when goal is
            City_Hospital; otherwise uses a uniform small estimate.

        Parameters:
            node (str): Current node.
            goal (str): Goal node.

        Returns:
            float: Non-negative heuristic estimate.
        """
        if goal == "City_Hospital":
            return self._heuristic_to_hospital.get(node, 5)
        # Generic: if goal is elsewhere, use 0 (A* degenerates to UCS)
        return 0

    def _astar(self, source, destination):
        """
        Function: _astar
        Description:
            A* Search on the weighted city graph. Combines the actual
            path cost g(n) with the heuristic h(n) to efficiently
            find the optimal path.

        Parameters:
            source      (str): Starting node.
            destination (str): Target node.

        Returns:
            tuple: (path: list[str], total_cost: float)
                   or (None, None) if no path exists.
        """
        # Priority queue: (f_cost, g_cost, node, path)
        # f_cost = g_cost + h_cost
        h_start = self._heuristic(source, destination)
        priority_queue = []
        heapq.heappush(priority_queue, (h_start, 0, source, [source]))

        visited = {}

        while priority_queue:
            f_cost, g_cost, current_node, path = heapq.heappop(priority_queue)

            if current_node in visited and visited[current_node] <= g_cost:
                continue
            visited[current_node] = g_cost

            if current_node == destination:
                return path, g_cost

            neighbors = self._weighted_graph.get(current_node, [])
            for neighbor, edge_weight in neighbors:
                new_g_cost = g_cost + edge_weight
                new_h_cost = self._heuristic(neighbor, destination)
                new_f_cost = new_g_cost + new_h_cost

                if neighbor not in visited or visited.get(neighbor, float("inf")) > new_g_cost:
                    heapq.heappush(
                        priority_queue,
                        (new_f_cost, new_g_cost, neighbor, path + [neighbor])
                    )

        return None, None  # No path found

    def get_graph_info(self):
        """
        Function: get_graph_info
        Description:
            Returns summary information about both city graph variants
            for display or debugging.

        Returns:
            dict: Summary of nodes and edge counts for each graph.
        """
        unweighted_edges = sum(
            len(neighbors) for neighbors in self._unweighted_graph.values()
        ) // 2

        weighted_edges = sum(
            len(neighbors) for neighbors in self._weighted_graph.values()
        ) // 2

        return {
            "total_nodes"           : len(self._unweighted_graph),
            "unweighted_edge_count" : unweighted_edges,
            "weighted_edge_count"   : weighted_edges,
            "nodes"                 : list(self._unweighted_graph.keys())
        }
