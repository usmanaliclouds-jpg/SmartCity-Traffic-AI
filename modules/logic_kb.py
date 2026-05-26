"""
Module 4: Logic / Knowledge Base Module
Description:
    This module is responsible for policy validation and rule-based
    reasoning. It checks whether a requested action is:
      - Allowed under current traffic policy.
      - Authorized given the vehicle type and zone.
      - Consistent with emergency priority rules.
      - Logically supported by the knowledge base.

    This module protects the system from unsafe or unauthorized
    actions. It is the gatekeeper before any constrained control
    allocation is executed.

    Predicates and Rules implemented (as per project spec):

    Predicates:
        Vehicle(v), EmergencyVehicle(v), CivilianVehicle(v)
        Location(l), SignalZone(z), Hospital(h)
        Request(req), RequestType(req, type)
        CurrentLocation(v,l), Destination(v,l)
        IncidentSeverity(v, level), TimeSensitive(v)
        Priority(v, level)
        Authorized(v, action), AllowedAction(v, action)
        EmergencyCorridor(v), EmergencyRoute(v)
        SignalOverride(z)
        Approved(v, req), Rejected(v, req)

    Rules:
        R1:  EmergencyVehicle(v) ^ IncidentSeverity(v,High)   -> Priority(v,Critical)
        R2:  EmergencyVehicle(v) ^ TimeSensitive(v)           -> Priority(v,High)
        R3:  CivilianVehicle(v)                               -> Priority(v,Normal)
        R4:  EmergencyVehicle(v) ^ SignalZone(z) -> Authorized(v, SignalOverride(z))
        R5:  CivilianVehicle(v)  ^ SignalZone(z) -> ~Authorized(v, SignalOverride(z))
        R6:  EmergencyVehicle(v) ^ Destination(v,h) ^ Hospital(h) -> EmergencyCorridor(v)
        R7:  EmergencyCorridor(v) -> Authorized(v, EmergencyRoute)
        R8:  Authorized(v,action) -> AllowedAction(v,action)
        R9:  ~AllowedAction(v,action) -> Rejected(v,req)
        R10: Priority(v,Critical) ^ ~Authorized(v,action) -> ~AllowedAction(v,action)
        R11: Priority(v,Critical) ^ Authorized(v,EmergencyRoute) -> AllowedAction(v,SignalOverride)
        R12: AllowedAction(v,action) -> Approved(v,req)
        R13: RequestType(req,Route_Request) -> Approved(v,req)
        R14: RequestType(req,Policy_Check) ^ Authorized(v,action) -> Approved(v,req)
        R15: RequestType(req,Policy_Check) ^ ~Authorized(v,action) -> Rejected(v,req)
        R16: RequestType(req,Control_Allocation_Request) ^ AllowedAction(v,action) -> Approved(v,req)
        R17: RequestType(req,Emergency_Response_Request) ^ Priority(v,level) ^
             Authorized(v,EmergencyRoute) -> Approved(v,req)
        R18: RequestType(req,Integrated_City_Service_Request) ^ Priority(v,Critical) ^
             Authorized(v,EmergencyRoute) ^ AllowedAction(v,action) -> Approved(v,req)
"""


class LogicKnowledgeBase:
    """
    Implements rule-based reasoning and policy validation for the
    Smart City Traffic system. Uses forward chaining over a set of
    domain predicates and rules.
    """

    # Locations designated as hospitals in the knowledge base
    HOSPITALS = {"City_Hospital"}

    # Locations that are signal-controlled zones
    SIGNAL_ZONES = {
        "Central_Junction", "North_Station",
        "East_Market", "River_Bridge", "City_Hospital"
    }

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the Logic/Knowledge Base module. Loads static
            facts about hospital locations and signal zones.
        """
        self._hospitals    = self.HOSPITALS
        self._signal_zones = self.SIGNAL_ZONES

    def validate(self, processed_request, ann_result=None):
        """
        Function: validate
        Description:
            Performs forward-chaining rule evaluation on the given
            request. Derives facts about priority, authorization,
            corridor eligibility, and final approval or rejection.

        Parameters:
            processed_request (dict): Validated and preprocessed request.
            ann_result        (dict): Output from ANN module (optional).
                                      If provided, ANN priority level
                                      is used to override derivation.

        Returns:
            dict: {
                'authorized'      : bool,
                'policy_status'   : str (APPROVED / REJECTED),
                'priority_level'  : str,
                'emergency_corridor': bool,
                'signal_override' : bool,
                'allowed_action'  : bool,
                'rules_fired'     : list[str],
                'justification'   : str
            }
        """
        req_type     = processed_request.get("request_category", "")
        vehicle_type = processed_request.get("vehicle_type", "civilian_car")
        severity     = processed_request.get("incident_severity", "Low")
        time_sens    = processed_request.get("time_sensitivity", False)
        destination  = processed_request.get("destination", "")
        control_zone = processed_request.get("control_zone", "")
        priority_claim = processed_request.get("priority_claim", False)
        is_emergency = processed_request.get("is_emergency_vehicle", False)

        rules_fired  = []
        facts        = {}

        # ---- Derive base vehicle classification ----
        facts["EmergencyVehicle"] = is_emergency
        facts["CivilianVehicle"]  = not is_emergency

        # ---- Derive Signal Zone fact ----
        facts["SignalZone"] = control_zone in self._signal_zones

        # ---- Derive Hospital destination fact ----
        facts["Hospital"] = destination in self._hospitals

        # ---- Rule R1: Emergency + High severity -> Critical priority ----
        if facts["EmergencyVehicle"] and severity in ("High", "Critical"):
            facts["Priority"] = "Critical"
            rules_fired.append("R1: EmergencyVehicle ^ IncidentSeverity(High/Critical) -> Priority(Critical)")
        # ---- Rule R2: Emergency + TimeSensitive -> High priority ----
        elif facts["EmergencyVehicle"] and time_sens:
            facts["Priority"] = "High"
            rules_fired.append("R2: EmergencyVehicle ^ TimeSensitive -> Priority(High)")
        # ---- Rule R3: Civilian -> Normal priority ----
        elif facts["CivilianVehicle"]:
            facts["Priority"] = "Normal"
            rules_fired.append("R3: CivilianVehicle -> Priority(Normal)")
        else:
            facts["Priority"] = "Low"

        # Override with ANN result if provided and it's more urgent
        if ann_result is not None:
            ann_level = ann_result.get("priority_level", "Low")
            priority_rank = {"Low": 0, "Normal": 1, "High": 2, "Critical": 3}
            if priority_rank.get(ann_level, 0) > priority_rank.get(facts["Priority"], 0):
                facts["Priority"] = ann_level
                rules_fired.append(f"ANN_OVERRIDE: ANN predicted {ann_level} -> Priority updated")

        # ---- Rule R4: Emergency + SignalZone -> Authorized(SignalOverride) ----
        if facts["EmergencyVehicle"] and facts["SignalZone"]:
            facts["Authorized_SignalOverride"] = True
            rules_fired.append("R4: EmergencyVehicle ^ SignalZone -> Authorized(SignalOverride)")
        # ---- Rule R5: Civilian + SignalZone -> ~Authorized(SignalOverride) ----
        elif facts["CivilianVehicle"] and facts["SignalZone"]:
            facts["Authorized_SignalOverride"] = False
            rules_fired.append("R5: CivilianVehicle ^ SignalZone -> ~Authorized(SignalOverride)")
        else:
            facts["Authorized_SignalOverride"] = facts["EmergencyVehicle"]

        # ---- Rule R6: Emergency + Destination=Hospital -> EmergencyCorridor ----
        if facts["EmergencyVehicle"] and facts["Hospital"]:
            facts["EmergencyCorridor"] = True
            rules_fired.append("R6: EmergencyVehicle ^ Hospital -> EmergencyCorridor")
        else:
            facts["EmergencyCorridor"] = False

        # ---- Rule R7: EmergencyCorridor -> Authorized(EmergencyRoute) ----
        if facts["EmergencyCorridor"]:
            facts["Authorized_EmergencyRoute"] = True
            rules_fired.append("R7: EmergencyCorridor -> Authorized(EmergencyRoute)")
        elif facts["EmergencyVehicle"]:
            facts["Authorized_EmergencyRoute"] = True  # Emergency vehicles always have route auth
        else:
            facts["Authorized_EmergencyRoute"] = False

        # Overall authorization flag
        facts["Authorized"] = (
            facts.get("Authorized_SignalOverride", False)
            or facts.get("Authorized_EmergencyRoute", False)
            or facts["EmergencyVehicle"]
        )

        # ---- Rule R8: Authorized -> AllowedAction ----
        if facts["Authorized"]:
            facts["AllowedAction"] = True
            rules_fired.append("R8: Authorized -> AllowedAction")
        else:
            facts["AllowedAction"] = False

        # ---- Rule R10: Priority(Critical) ^ ~Authorized -> ~AllowedAction ----
        if facts["Priority"] == "Critical" and not facts["Authorized"]:
            facts["AllowedAction"] = False
            rules_fired.append("R10: Priority(Critical) ^ ~Authorized -> ~AllowedAction")

        # ---- Rule R11: Priority(Critical) ^ Authorized(EmergencyRoute) -> AllowedAction(SignalOverride) ----
        if facts["Priority"] == "Critical" and facts.get("Authorized_EmergencyRoute", False):
            facts["AllowedAction"] = True
            facts["Authorized_SignalOverride"] = True
            rules_fired.append("R11: Priority(Critical) ^ Authorized(EmergencyRoute) -> AllowedAction(SignalOverride)")

        # ---- Rule R9: ~AllowedAction -> Rejected ----
        if not facts["AllowedAction"]:
            rules_fired.append("R9: ~AllowedAction -> Rejected")

        # ---- Determine Approval based on request type ----
        approved = self._evaluate_approval(req_type, facts, rules_fired)

        # Build justification text
        justification = self._build_justification(facts, approved, req_type)

        return {
            "authorized"          : facts["Authorized"],
            "policy_status"       : "APPROVED" if approved else "REJECTED",
            "priority_level"      : facts["Priority"],
            "emergency_corridor"  : facts["EmergencyCorridor"],
            "signal_override"     : facts.get("Authorized_SignalOverride", False),
            "allowed_action"      : facts["AllowedAction"],
            "rules_fired"         : rules_fired,
            "justification"       : justification
        }

    def _evaluate_approval(self, req_type, facts, rules_fired):
        """
     
        Function: _evaluate_approval
        Description:
            Evaluates the approval outcome based on request type and
            derived facts. Applies rules R12 through R18.

        Parameters:
            req_type   (str):  Request category.
            facts      (dict): Derived logical facts.
            rules_fired(list): List to append fired rule descriptions.

        Returns:
            bool: True if request is approved, False if rejected.
       
        """
        # R13: Route_Request always approved
        if req_type == "Route_Request":
            rules_fired.append("R13: RequestType(Route_Request) -> Approved")
            return True

        # R14/R15: Policy_Check depends on authorization
        if req_type == "Policy_Check":
            if facts["Authorized"]:
                rules_fired.append("R14: Policy_Check ^ Authorized -> Approved")
                return True
            else:
                rules_fired.append("R15: Policy_Check ^ ~Authorized -> Rejected")
                return False

        # R16: Control_Allocation_Request depends on AllowedAction
        if req_type == "Control_Allocation_Request":
            if facts["AllowedAction"]:
                rules_fired.append("R16: Control_Allocation ^ AllowedAction -> Approved")
                return True
            else:
                rules_fired.append("R16: Control_Allocation ^ ~AllowedAction -> Rejected")
                return False

        # R17: Emergency_Response_Request: Priority + AuthorizedEmergencyRoute
        if req_type == "Emergency_Response_Request":
            if facts.get("Authorized_EmergencyRoute", False):
                rules_fired.append("R17: Emergency_Response ^ Priority ^ Authorized(EmergencyRoute) -> Approved")
                return True
            elif facts["AllowedAction"]:
                return True
            else:
                return False

        # R18: Integrated request: Critical priority + Route + AllowedAction
        if req_type == "Integrated_City_Service_Request":
            if (facts["Priority"] in ("Critical", "High")
                    and facts.get("Authorized_EmergencyRoute", False)
                    and facts["AllowedAction"]):
                rules_fired.append("R18: Integrated ^ Critical ^ AuthRoute ^ AllowedAction -> Approved")
                return True
            elif facts["AllowedAction"]:
                return True
            else:
                return False

        # Default: if AllowedAction, approve
        if facts["AllowedAction"]:
            rules_fired.append("R12: AllowedAction -> Approved")
            return True

        return False

    def _build_justification(self, facts, approved, req_type):
        """
        Function: _build_justification
        Description:
            Constructs a human-readable justification string
            summarizing the reasoning outcome.

        Parameters:
            facts    (dict): Derived logical facts.
            approved (bool): Whether the request was approved.
            req_type (str):  Request category.

        Returns:
            str: Justification text.
        """
        vehicle_class = "emergency vehicle" if facts["EmergencyVehicle"] else "civilian vehicle"
        priority      = facts["Priority"]
        authorized    = facts["Authorized"]
        corridor      = facts["EmergencyCorridor"]
        sig_override  = facts.get("Authorized_SignalOverride", False)
        outcome       = "APPROVED" if approved else "REJECTED"

        parts = [
            f"Request {outcome} for {vehicle_class}.",
            f"Priority level: {priority}.",
            f"Authorization status: {'Authorized' if authorized else 'Not Authorized'}.",
        ]
        if corridor:
            parts.append("Emergency corridor granted (destination is a hospital).")
        if sig_override:
            parts.append("Signal override authorized in control zone.")
        if not authorized:
            parts.append(
                "Action rejected: civilian vehicles are not authorized for "
                "signal override or emergency routing."
            )

        return " ".join(parts)
