"""
Module 7: Final Response Layer
Description:
    The Final Response Layer aggregates the outputs of all modules
    that were actually used for the current request. It presents
    only the relevant fields — no unused module outputs are shown.

    For example:
      - A Route_Request will only show the route and travel estimate.
      - An Emergency_Response_Request will show priority level,
        corridor status, signal plan, and route.
      - A Policy_Check will show authorization and policy reasoning.

    The response is selective, operationally meaningful, and
    explainable — as required by the project specification.

    Possible response components:
        - Recommended route
        - Estimated travel / delay info
        - Predicted priority level
        - Policy validation result
        - Assigned signal-control / lane-control action
        - Decision message and explanatory text
"""


class FinalResponseLayer:
    """
    Aggregates and formats the final output of the Smart City Traffic
    & Emergency Response AI System pipeline.
    """

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the FinalResponseLayer. No external
            dependencies are required at construction time.
        """
        pass

    def generate(self, results):
        """
        Function: generate
        Description:
            Builds the final response dictionary from all module
            outputs present in 'results'. Only includes fields that
            were produced by modules that actually ran.

        Parameters:
            results (dict): Accumulated pipeline results containing:
                - 'request'      : processed request dict
                - 'pipeline'     : list of module names that ran
                - 'ann_result'   : ANN output or None
                - 'logic_result' : Logic/KB output or None
                - 'csp_result'   : CSP output or None
                - 'search_result': Search output or None

        Returns:
            dict: Final structured response with only relevant fields.
        """
        request       = results.get("request", {})
        pipeline      = results.get("pipeline", [])
        ann_result    = results.get("ann_result")
        logic_result  = results.get("logic_result")
        csp_result    = results.get("csp_result")
        search_result = results.get("search_result")

        response = {
            "request_id"       : request.get("request_id", "N/A"),
            "request_category" : request.get("request_category", "N/A"),
            "vehicle_type"     : request.get("vehicle_type", "N/A"),
            "source"           : request.get("current_location", "N/A"),
            "destination"      : request.get("destination", "N/A"),
            "pipeline_used"    : pipeline,
            "status"           : "SUCCESS",
            "components"       : {}
        }

        # ---- ANN Component ----
        if ann_result is not None:
            response["components"]["priority"] = {
                "priority_level" : ann_result.get("priority_level", "N/A"),
                "urgency_score"  : ann_result.get("urgency_score", 0.0),
                "is_urgent"      : ann_result.get("is_urgent", False),
                "class_probs"    : ann_result.get("class_probs", [])
            }

        # ---- Logic/KB Component ----
        if logic_result is not None:
            policy_status = logic_result.get("policy_status", "N/A")
            if policy_status == "REJECTED":
                response["status"] = "REJECTED"

            response["components"]["policy"] = {
                "policy_status"      : policy_status,
                "authorized"         : logic_result.get("authorized", False),
                "emergency_corridor" : logic_result.get("emergency_corridor", False),
                "signal_override"    : logic_result.get("signal_override", False),
                "justification"      : logic_result.get("justification", ""),
                "rules_fired"        : logic_result.get("rules_fired", [])
            }

        # ---- CSP Component ----
        if csp_result is not None:
            response["components"]["signal_control"] = {
                "assigned_action" : csp_result.get("assigned_action", "N/A"),
                "feasible"        : csp_result.get("feasible", False),
                "signal_sequence" : csp_result.get("signal_sequence", {}),
                "control_zone"    : csp_result.get("control_zone", "N/A"),
                "corridor_active" : csp_result.get("corridor_active", False),
                "reason"          : csp_result.get("reason", "")
            }

        # ---- Search/Navigation Component ----
        if search_result is not None:
            if not search_result.get("found", False):
                response["status"] = "PARTIAL"

            response["components"]["navigation"] = {
                "route_found"  : search_result.get("found", False),
                "algorithm"    : search_result.get("algorithm", "N/A"),
                "path"         : search_result.get("path", []),
                "total_cost"   : search_result.get("total_cost"),
                "hops"         : search_result.get("hops", 0),
                "reason"       : search_result.get("reason", "")
            }

        # ---- Build summary message ----
        response["summary"] = self._build_summary(response, request)

        return response

    def _build_summary(self, response, request):
        """
        Function: _build_summary
        Description:
            Constructs a plain-language summary of the final decision
            for display to system operators or city users.

        Parameters:
            response (dict): The partially built response dictionary.
            request  (dict): The original processed request.

        Returns:
            str: A multi-line human-readable decision summary.
        """
        lines = []
        category    = response.get("request_category", "")
        source      = response.get("source", "")
        destination = response.get("destination", "")
        status      = response.get("status", "SUCCESS")
        components  = response.get("components", {})

        lines.append(f"Request Category : {category}")
        lines.append(f"Vehicle Type     : {response.get('vehicle_type', 'N/A')}")
        lines.append(f"From             : {source}  -->  {destination}")
        lines.append(f"Overall Status   : {status}")
        lines.append("")

        # Priority summary
        if "priority" in components:
            p = components["priority"]
            lines.append(
                f"Priority Level   : {p['priority_level']} "
                f"(Urgency Score: {p['urgency_score']})"
            )
            lines.append(f"Urgent           : {'Yes' if p['is_urgent'] else 'No'}")
            lines.append("")

        # Policy summary
        if "policy" in components:
            pol = components["policy"]
            lines.append(f"Policy Status    : {pol['policy_status']}")
            lines.append(f"Authorized       : {'Yes' if pol['authorized'] else 'No'}")
            if pol["emergency_corridor"]:
                lines.append("Emergency Corridor : ACTIVE (destination is a hospital)")
            if pol["signal_override"]:
                lines.append("Signal Override    : GRANTED")
            lines.append(f"Justification    : {pol['justification']}")
            lines.append("")

        # Signal control summary
        if "signal_control" in components:
            sc = components["signal_control"]
            lines.append(f"Control Action   : {sc['assigned_action']}")
            lines.append(f"CSP Feasible     : {'Yes' if sc['feasible'] else 'No'}")
            if sc.get("signal_sequence"):
                lines.append("Signal Plan      :")
                for intersection, phase in sc["signal_sequence"].items():
                    lines.append(f"    {intersection:<25} -> {phase}")
            lines.append("")

        # Navigation summary
        if "navigation" in components:
            nav = components["navigation"]
            if nav["route_found"]:
                path_str = " -> ".join(nav["path"])
                lines.append(f"Algorithm Used   : {nav['algorithm']}")
                lines.append(f"Recommended Route: {path_str}")
                lines.append(f"Total Hops       : {nav['hops']}")
                if nav["total_cost"] is not None:
                    lines.append(f"Estimated Cost   : {nav['total_cost']} units")
            else:
                lines.append(f"Route Status     : No route found.")
                if nav.get("reason"):
                    lines.append(f"Reason           : {nav['reason']}")
            lines.append("")

        return "\n".join(lines)

    def display(self, response):
        """
        Function: display
        Description:
            Prints the final response to the console in a clean,
            formatted layout suitable for operator viewing.

        Parameters:
            response (dict): The final response dictionary from generate().
        """
        print(f"\n  Request ID : {response.get('request_id', 'N/A')}")
        print(f"  Status     : {response.get('status', 'N/A')}")
        print()

        summary = response.get("summary", "No summary available.")
        for line in summary.split("\n"):
            print(f"  {line}")

        pipeline = response.get("pipeline_used", [])
        print(f"  Modules Run: {' -> '.join(pipeline)}")
        print()
