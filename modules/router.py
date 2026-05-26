"""
Module 2: Request Router
Description:
    The Request Router is the control-flow manager of the system.
    It receives a validated, preprocessed request and determines
    which processing modules should be executed and in what order.

    The router prevents inappropriate module calls, preserves
    sequencing rules, and ensures each request is handled according
    to its operational need rather than a one-size-fits-all pipeline.

    Pipeline mappings by category:
      - Route_Request              -> Search
      - Policy_Check               -> Logic_KB
      - Control_Allocation_Request -> Logic_KB -> CSP
      - Emergency_Response_Request -> ANN -> Logic_KB -> CSP -> Search
      - Integrated_City_Service_Request -> ANN -> Logic_KB -> CSP ->
                                           Search -> Final_Response

    All pipelines end with Final_Response (always applied).
"""


class RequestRouter:
    """
    Determines the correct processing pipeline for each request
    category and returns an ordered list of module names to execute.
    """

    # Pipeline map: category -> ordered list of AI modules to run
    PIPELINE_MAP = {
        "Route_Request": [
            "Search"
        ],
        "Policy_Check": [
            "Logic_KB"
        ],
        "Control_Allocation_Request": [
            "Logic_KB",
            "CSP"
        ],
        "Emergency_Response_Request": [
            "ANN",
            "Logic_KB",
            "CSP",
            "Search"
        ],
        "Integrated_City_Service_Request": [
            "ANN",
            "Logic_KB",
            "CSP",
            "Search",
            "Final_Response"
        ]
    }

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the RequestRouter. Loads the pipeline
            configuration map.
        """
        self._pipeline_map = self.PIPELINE_MAP

    def route(self, processed_request):
        """
        Function: route
        Description:
            Accepts a validated processed request and returns the
            ordered list of AI modules to invoke for that request.

        Parameters:
            processed_request (dict): Validated request from preprocessor.

        Returns:
            list[str]: Ordered list of module names to execute.
                       Raises ValueError if category is unrecognized.
        """
        category = processed_request.get("request_category", "")

        if not category:
            raise ValueError(
                "RequestRouter: 'request_category' is missing from the request. "
                "Cannot determine processing pipeline."
            )

        pipeline = self._pipeline_map.get(category)

        if pipeline is None:
            raise ValueError(
                f"RequestRouter: Unrecognized request category '{category}'. "
                f"Valid categories are: {list(self._pipeline_map.keys())}"
            )

        # Always ensure Final_Response is included at the end
        pipeline_steps = list(pipeline)
        if "Final_Response" not in pipeline_steps:
            pipeline_steps.append("Final_Response")

        return pipeline_steps

    def get_all_pipelines(self):
        """
        Function: get_all_pipelines
        Description:
            Returns the full pipeline configuration map for inspection
            or debugging purposes.

        Returns:
            dict: Mapping of category name to list of module steps.
        """
        return dict(self._pipeline_map)

    def describe_pipeline(self, category):
        """
        Function: describe_pipeline
        Description:
            Returns a human-readable description of the pipeline
            associated with a given request category.

        Parameters:
            category (str): The request category name.

        Returns:
            str: A formatted pipeline description string.
        """
        pipeline = self._pipeline_map.get(category)
        if pipeline is None:
            return f"No pipeline found for category: '{category}'"

        steps = pipeline + (["Final_Response"] if "Final_Response" not in pipeline else [])
        return f"[{category}] -> " + " -> ".join(steps)
