"""
Module 1: Input & Preprocessing Module
Description:
    This module is the entry point of the system. It receives raw
    traffic request data, validates all required fields, normalizes
    values to system-standard formats, and builds a clean internal
    request object.

    For requests requiring ANN-based priority prediction, this
    module also prepares the feature vector used by the ANN module.

    Key responsibilities:
      - Validate required fields are present and non-empty.
      - Normalize vehicle types, locations, severity, and categories.
      - Map categorical values to numeric form for ANN input.
      - Enforce that source != destination.
      - Flag invalid requests with descriptive error messages.

    This module does NOT process free-text commands (no NLP).
    All inputs must be structured and pre-categorized.
"""


class InputPreprocessor:
    """
    Validates, normalizes, and standardizes incoming traffic requests.
    Produces a clean internal request object or flags errors.
    """

    # Valid values for categorical fields
    VALID_CATEGORIES = [
        "Route_Request",
        "Policy_Check",
        "Control_Allocation_Request",
        "Emergency_Response_Request",
        "Integrated_City_Service_Request"
    ]

    VALID_VEHICLE_TYPES = [
        "ambulance", "fire_truck", "police",
        "civilian_car", "bus"
    ]

    EMERGENCY_VEHICLES = {"ambulance", "fire_truck", "police"}

    VALID_SEVERITIES = ["Low", "Medium", "High", "Critical"]

    VALID_DENSITIES = ["Low", "Medium", "High"]

    VALID_LOCATIONS = [
        "Police_HQ", "Traffic_Control_Center", "North_Station",
        "River_Bridge", "Stadium", "East_Market", "Central_Junction",
        "West_Terminal", "Airport_Road", "South_Residential",
        "City_Hospital", "Industrial_Zone", "Fire_Station"
    ]

    VALID_CONTROL_ZONES = [
        "Central_Junction", "North_Station", "East_Market",
        "River_Bridge", "City_Hospital"
    ]

    # Numeric encoding for ANN feature vector
    SEVERITY_ENCODING = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Critical": 1.00}
    DENSITY_ENCODING  = {"Low": 0.33, "Medium": 0.66, "High": 1.00}
    VEHICLE_ENCODING  = {
        "ambulance": 1.0, "fire_truck": 0.9, "police": 0.8,
        "civilian_car": 0.3, "bus": 0.4
    }

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the InputPreprocessor with required field
            definitions and validation rules.
        """
        # Required fields that must always be present
        self._required_fields = [
            "request_id", "request_category", "vehicle_type",
            "current_location", "destination"
        ]

        # Fields needed for advanced modules (ANN, Logic, CSP)
        self._advanced_fields = [
            "incident_severity", "time_sensitivity",
            "traffic_density", "priority_claim", "control_zone"
        ]

    def process(self, raw_data):
        """
        Function: process
        Description:
            Main processing function. Validates and normalizes the raw
            input dictionary, then returns a clean processed request.

        Parameters:
            raw_data (dict): Raw input from user/system.

        Returns:
            dict: Processed request with 'valid' flag, errors list,
                  normalized fields, and ANN feature vector (if needed).
        """
        errors = []

        # --- Validate required fields ---
        for field in self._required_fields:
            if field not in raw_data or raw_data[field] is None:
                errors.append(f"Missing required field: '{field}'")
            elif isinstance(raw_data[field], str) and raw_data[field].strip() == "":
                errors.append(f"Field '{field}' must not be empty.")

        if errors:
            return {"valid": False, "errors": errors}

        # --- Normalize and validate category ---
        category = raw_data["request_category"].strip()
        if category not in self.VALID_CATEGORIES:
            errors.append(
                f"Invalid request_category '{category}'. "
                f"Must be one of: {self.VALID_CATEGORIES}"
            )

        # --- Normalize vehicle type ---
        vehicle_type = raw_data["vehicle_type"].strip().lower()
        if vehicle_type not in self.VALID_VEHICLE_TYPES:
            errors.append(
                f"Invalid vehicle_type '{vehicle_type}'. "
                f"Must be one of: {self.VALID_VEHICLE_TYPES}"
            )

        # --- Normalize locations ---
        current_location = self._normalize_location(raw_data["current_location"])
        destination = self._normalize_location(raw_data["destination"])

        if current_location not in self.VALID_LOCATIONS:
            errors.append(f"Invalid current_location: '{current_location}'.")
        if destination not in self.VALID_LOCATIONS:
            errors.append(f"Invalid destination: '{destination}'.")

        if current_location == destination and current_location in self.VALID_LOCATIONS:
            errors.append("current_location and destination must be different.")

        # --- Normalize advanced fields ---
        severity = self._get_field(raw_data, "incident_severity", "Low")
        time_sensitive = self._get_bool_field(raw_data, "time_sensitivity", False)
        traffic_density = self._get_field(raw_data, "traffic_density", "Low")
        priority_claim = self._get_bool_field(raw_data, "priority_claim", False)
        control_zone = self._get_field(raw_data, "control_zone", "Central_Junction")
        description_note = raw_data.get("description_note", "").strip()

        if severity not in self.VALID_SEVERITIES:
            errors.append(f"Invalid incident_severity '{severity}'.")
        if traffic_density not in self.VALID_DENSITIES:
            errors.append(f"Invalid traffic_density '{traffic_density}'.")
        if control_zone not in self.VALID_CONTROL_ZONES:
            errors.append(f"Invalid control_zone '{control_zone}'.")

        if errors:
            return {"valid": False, "errors": errors}

        # Determine if vehicle is classified as emergency
        is_emergency_vehicle = vehicle_type in self.EMERGENCY_VEHICLES

        # Build clean request object
        processed_request = {
            "valid": True,
            "errors": [],
            "request_id": raw_data["request_id"],
            "request_category": category,
            "vehicle_type": vehicle_type,
            "is_emergency_vehicle": is_emergency_vehicle,
            "current_location": current_location,
            "destination": destination,
            "incident_severity": severity,
            "time_sensitivity": time_sensitive,
            "traffic_density": traffic_density,
            "priority_claim": priority_claim,
            "control_zone": control_zone,
            "description_note": description_note,
            "ann_feature_vector": self._build_ann_features(
                vehicle_type, severity, time_sensitive,
                traffic_density, priority_claim, current_location, destination
            )
        }

        return processed_request

    def _normalize_location(self, location_str):
        """
        Function: _normalize_location
        Description:
            Strips whitespace and attempts to match the location string
            to a known valid location (case-insensitive).

        Parameters:
            location_str (str): Raw location string.

        Returns:
            str: Normalized location name, or original string if no
                 match is found.
        """
        loc = location_str.strip()
        # Case-insensitive match
        for valid_loc in self.VALID_LOCATIONS:
            if loc.lower() == valid_loc.lower():
                return valid_loc
        return loc  # return as-is; will fail validation below

    def _get_field(self, data, key, default):
        """
        Function: _get_field
        Description:
            Safely retrieves a string field from data dictionary with
            a fallback default value.

        Parameters:
            data    (dict): The input data dictionary.
            key     (str):  Field name to retrieve.
            default (str):  Default value if field is missing/None.

        Returns:
            str: The field value or default.
        """
        value = data.get(key, default)
        if value is None:
            return default
        return str(value).strip() if isinstance(value, str) else value

    def _get_bool_field(self, data, key, default):
        """
        Function: _get_bool_field
        Description:
            Safely retrieves a boolean field from data. Handles string
            representations like 'yes', 'true', '1'.

        Parameters:
            data    (dict): The input data dictionary.
            key     (str):  Field name to retrieve.
            default (bool): Default boolean value.

        Returns:
            bool: Parsed boolean value.
        """
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("yes", "true", "1")
        return bool(value)

    def _build_ann_features(self, vehicle_type, severity, time_sensitive,
                             traffic_density, priority_claim,
                             current_location, destination):
        """
        Function: _build_ann_features
        Description:
            Constructs a numeric feature vector for ANN priority
            prediction from categorical and boolean input fields.

            Feature vector:
                [vehicle_urgency, severity_score, time_sensitive_flag,
                 density_score, priority_claim_flag, distance_estimate]

        Parameters:
            vehicle_type    (str):  Vehicle type string.
            severity        (str):  Incident severity level.
            time_sensitive  (bool): Whether request is time-sensitive.
            traffic_density (str):  Traffic density level.
            priority_claim  (bool): Whether priority is claimed.
            current_location(str):  Source node.
            destination     (str):  Destination node.

        Returns:
            list[float]: Normalized feature vector of length 6.
   
        """
        vehicle_score   = self.VEHICLE_ENCODING.get(vehicle_type, 0.3)
        severity_score  = self.SEVERITY_ENCODING.get(severity, 0.25)
        time_score      = 1.0 if time_sensitive else 0.0
        density_score   = self.DENSITY_ENCODING.get(traffic_density, 0.33)
        priority_score  = 1.0 if priority_claim else 0.0

        # Simplified distance estimate: ratio of location indices as proxy
        all_locs = self.VALID_LOCATIONS
        src_idx  = all_locs.index(current_location) if current_location in all_locs else 0
        dst_idx  = all_locs.index(destination) if destination in all_locs else 0
        distance_estimate = abs(dst_idx - src_idx) / max(len(all_locs) - 1, 1)

        return [
            vehicle_score,
            severity_score,
            time_score,
            density_score,
            priority_score,
            distance_estimate
        ]
