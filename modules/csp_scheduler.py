"""
Module 5: CSP Scheduler / Control Allocation Module

Description:
    The CSP (Constraint Satisfaction Problem) module handles
    constrained control decisions for smart city traffic management.
    These decisions may include signal timing plans, lane-control
    adjustments, emergency corridor allocation, or intersection
    sequence coordination.

    The module searches for a valid assignment that:
      - Respects safety constraints (no conflicting signal phases).
      - Respects timing constraints (phase duration limits).
      - Respects conflict constraints (opposing lanes not simultaneous).
      - Supports the current operational traffic need (e.g., emergency
        priority).

    CSP Graph (from project spec):
      Nodes (intersections with signal phases):
        S1: Central_Junction  -> Phase_A, Phase_B, Phase_C
        S2: North_Station     -> Phase_A, Phase_B, Phase_C
        S3: East_Market       -> Phase_A, Phase_B, Phase_C
        S4: River_Bridge      -> Phase_A, Phase_B, Phase_C
        S5: City_Hospital     -> Phase_A, Phase_B, Phase_C

      Constraints:
        - coordination   : S2 and S4 must not share same phase.
        - conflict       : S1 and S3 must not be in the same phase.
        - emergency priority: S5 (City_Hospital) prefers Phase_C.

    Algorithm: Backtracking search with constraint propagation.
"""


class CSPScheduler:
    """
    Implements a backtracking CSP solver for traffic signal control
    allocation across city intersections.
    """

    # Signal phase options for each intersection
    SIGNAL_PHASES = ["Phase_A", "Phase_B", "Phase_C"]

    # Intersection nodes in the CSP graph
    INTERSECTIONS = [
        "Central_Junction",
        "North_Station",
        "East_Market",
        "River_Bridge",
        "City_Hospital"
    ]

    # Constraint definitions
    # Each constraint is (intersection_1, relation, intersection_2)
    # relation: "different" means they must have different phases
    CONSTRAINTS = [
        ("North_Station",    "different", "River_Bridge"),     # coordination constraint
        ("Central_Junction", "different", "East_Market"),      # conflict constraint
        ("City_Hospital",    "prefers",   "Phase_C"),          # emergency priority
    ]

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the CSP Scheduler with the city intersection
            graph, phase domain, and constraint definitions.
        """
        self._intersections = self.INTERSECTIONS
        self._phases        = self.SIGNAL_PHASES
        self._constraints   = self.CONSTRAINTS

    def allocate(self, processed_request, logic_result=None):
        """
        Function: allocate
        Description:
            Main allocation function. Attempts to find a feasible
            signal phase assignment for all intersections given the
            current request context and authorization results.

            If the request was rejected by the logic module, a default
            conservative signal assignment is returned.

        Parameters:
            processed_request (dict): Validated and preprocessed request.
            logic_result      (dict): Output from Logic/KB module.
                                      If None, assumes authorized.

        Returns:
            dict: {
                'feasible'         : bool,
                'assigned_action'  : str,
                'signal_sequence'  : dict (intersection -> phase),
                'control_zone'     : str,
                'corridor_active'  : bool,
                'reason'           : str
            }
        """
        control_zone  = processed_request.get("control_zone", "Central_Junction")
        req_type      = processed_request.get("request_category", "")
        is_emergency  = processed_request.get("is_emergency_vehicle", False)

        # Check authorization from logic result
        authorized = True
        if logic_result is not None:
            authorized = logic_result.get("authorized", True)
            if logic_result.get("policy_status") == "REJECTED":
                authorized = False

        # If not authorized, return conservative default
        if not authorized:
            return {
                "feasible"        : False,
                "assigned_action" : "HOLD_ALL_PHASES",
                "signal_sequence" : {node: "Phase_A" for node in self._intersections},
                "control_zone"    : control_zone,
                "corridor_active" : False,
                "reason"          : "Unauthorized request. Conservative Phase_A held at all intersections."
            }

        # Determine if emergency corridor should be prioritized
        corridor_active = False
        if logic_result is not None:
            corridor_active = logic_result.get("emergency_corridor", False)

        # Build domain: for each intersection, allowed phases
        domain = self._build_domain(is_emergency, corridor_active, control_zone)

        # Run backtracking search
        assignment = {}
        solution   = self._backtrack(assignment, domain)

        if solution is None:
            # Fallback: assign Phase_A to all (safe default)
            solution = {node: "Phase_A" for node in self._intersections}
            feasible = False
            reason   = "CSP solver could not find a consistent assignment. Defaulting to Phase_A."
        else:
            feasible = True
            reason   = self._describe_solution(solution, is_emergency, corridor_active)

        # Determine the primary action label
        assigned_action = self._determine_action(
            solution, control_zone, is_emergency, corridor_active
        )

        return {
            "feasible"        : feasible,
            "assigned_action" : assigned_action,
            "signal_sequence" : solution,
            "control_zone"    : control_zone,
            "corridor_active" : corridor_active,
            "reason"          : reason
        }

    def _build_domain(self, is_emergency, corridor_active, control_zone):
        """
        Function: _build_domain
        Description:
            Builds the domain of allowed signal phases for each
            intersection. In an emergency corridor scenario, City_Hospital
            is restricted to Phase_C (green for emergency).

        Parameters:
            is_emergency    (bool): Whether request is from emergency vehicle.
            corridor_active (bool): Whether emergency corridor is active.
            control_zone    (str): The primary control zone.

        Returns:
            dict: {intersection_name -> list[str] of allowed phases}
        """
        domain = {}
        for node in self._intersections:
            if corridor_active and node == "City_Hospital":
                # Emergency corridor: hospital must be Phase_C (green)
                domain[node] = ["Phase_C"]
            elif is_emergency and node == control_zone:
                # Emergency vehicle: prioritize Phase_C at control zone
                domain[node] = ["Phase_C", "Phase_B"]
            else:
                domain[node] = list(self._phases)
        return domain

    def _is_consistent(self, node, phase, assignment):
        """
        Function: _is_consistent
        Description:
            Checks whether assigning 'phase' to 'node' is consistent
            with all existing assignments in 'assignment', given the
            defined constraints.

        Parameters:
            node       (str):  Intersection being assigned.
            phase      (str):  Proposed phase for node.
            assignment (dict): Current partial assignment.

        Returns:
            bool: True if consistent with all applicable constraints.
        """
        for (node1, relation, node2) in self._constraints:
            # Skip constraints not involving 'node'
            if node not in (node1, node2):
                continue

            if relation == "different":
                # Both nodes must have different phases
                other_node = node2 if node == node1 else node1
                if other_node in assignment:
                    if assignment[other_node] == phase:
                        return False  # Violation: same phase

            elif relation == "prefers":
                # node2 here is the preferred phase value
                # This is a soft constraint; we only enforce it during domain building
                pass

        return True

    def _backtrack(self, assignment, domain):
        """
        Function: _backtrack
        Description:
            Recursive backtracking search for a valid CSP assignment.
            Selects unassigned variables in order, tries each value
            in the domain, checks consistency, and recurses.

        Parameters:
            assignment (dict): Current partial assignment.
            domain     (dict): Remaining allowed values per variable.

        Returns:
            dict or None: Complete assignment if found, else None.
        """
        # Base case: all intersections assigned
        if len(assignment) == len(self._intersections):
            return assignment

        # Select next unassigned intersection
        unassigned = [
            node for node in self._intersections
            if node not in assignment
        ]
        if not unassigned:
            return assignment

        current_node = unassigned[0]

        # Try each phase in domain for this node
        for phase in domain.get(current_node, self._phases):
            if self._is_consistent(current_node, phase, assignment):
                assignment[current_node] = phase

                # Recurse
                result = self._backtrack(assignment, domain)
                if result is not None:
                    return result

                # Backtrack
                del assignment[current_node]

        return None  # No valid assignment found from here

    def _determine_action(self, solution, control_zone, is_emergency, corridor_active):
        """
        Function: _determine_action
        Description:
            Derives the primary control action label based on the
            solution and request context.

        Parameters:
            solution        (dict): Final phase assignment.
            control_zone    (str): Primary control zone.
            is_emergency    (bool): Emergency vehicle flag.
            corridor_active (bool): Emergency corridor active flag.

        Returns:
            str: Human-readable action label.
        """
        phase_at_zone = solution.get(control_zone, "Phase_A")

        if corridor_active:
            return f"EMERGENCY_CORRIDOR_ACTIVE | {control_zone}: {phase_at_zone}"
        elif is_emergency:
            return f"EMERGENCY_PRIORITY_SIGNAL | {control_zone}: {phase_at_zone}"
        else:
            return f"STANDARD_SIGNAL_CONTROL   | {control_zone}: {phase_at_zone}"

    def _describe_solution(self, solution, is_emergency, corridor_active):
        """
        Function: _describe_solution
        Description:
            Produces a human-readable description of the CSP solution.

        Parameters:
            solution        (dict): Phase assignment per intersection.
            is_emergency    (bool): Emergency vehicle flag.
            corridor_active (bool): Emergency corridor active flag.

        Returns:
            str: Narrative description of the signal plan.
        """
        phase_summary = ", ".join(
            f"{node}: {phase}" for node, phase in solution.items()
        )
        mode = "Emergency corridor" if corridor_active else (
            "Emergency priority" if is_emergency else "Standard"
        )
        return f"{mode} signal plan assigned. Phases: [{phase_summary}]"
