"""
Module 3: ANN Priority Module

Description:
    The ANN (Artificial Neural Network) module estimates urgency
    or priority level for requests that require priority-aware
    handling. This is especially important for emergency response.

    The module does NOT decide whether an action is legally allowed.
    It predicts HOW URGENT a situation appears, based on structured
    operational indicators.

    Architecture:
        Binary Classifier:
            Input Layer (6 features) -> Sigmoid Activation -> Urgent/Not Urgent

        MLP Multi-class Classifier:
            Input Layer (6 features)
              -> Hidden Layer 1 (4 neurons, ReLU)
              -> Hidden Layer 2 (4 neurons, ReLU)
              -> Output Layer  (4 neurons, Softmax)
                 [Low, Normal, High, Critical]

    Features used:
        1. Vehicle urgency score
        2. Incident severity score
        3. Time sensitivity flag
        4. Traffic density score
        5. Priority claim flag
        6. Estimated route distance

    Implementation note:
        Weights are manually initialized to demonstrate priority
        estimation behavior (academic simulation). Real-world
        deployment would train these weights on historical data.

"""

import math


class ANNPriorityModule:
    """
    Implements a simplified MLP-based priority classifier for traffic
    request urgency estimation. Uses manually configured weights for
    simulation purposes.
    """

    # Priority level labels (output classes)
    PRIORITY_LEVELS = ["Low", "Normal", "High", "Critical"]

    # Urgency threshold for binary classifier
    BINARY_URGENCY_THRESHOLD = 0.5

    def __init__(self):
        """
        Function: __init__
        Description:
            Initializes the ANN module with manually prepared weight
            matrices for both the binary classifier and the MLP
            multi-class classifier. Bias terms are also defined.

            Architecture:
                Input size : 6
                Hidden L1  : 4 neurons
                Hidden L2  : 4 neurons
                Output     : 4 classes (Low, Normal, High, Critical)

        """
        # ---- Binary Classifier Weights (input -> 1 output) ----
        # Each weight reflects the feature's contribution to urgency
        self._binary_weights = [0.45, 0.35, 0.25, 0.20, 0.30, 0.10]
        self._binary_bias    = -0.80

        # ---- MLP Weights: Input (6) -> Hidden Layer 1 (4) ----
        # Shape: (4 neurons) x (6 inputs)
        self._w1 = [
            [0.50,  0.40,  0.30,  0.20,  0.35,  0.10],  # neuron 1
            [0.30,  0.50,  0.20,  0.40,  0.25,  0.15],  # neuron 2
            [0.20,  0.30,  0.50,  0.35,  0.40,  0.20],  # neuron 3
            [0.40,  0.20,  0.35,  0.50,  0.30,  0.25],  # neuron 4
        ]
        self._b1 = [-0.60, -0.55, -0.50, -0.45]

        # ---- MLP Weights: Hidden Layer 1 (4) -> Hidden Layer 2 (4) ----
        self._w2 = [
            [0.60,  0.20,  0.30,  0.40],  # neuron 1
            [0.20,  0.60,  0.40,  0.30],  # neuron 2
            [0.30,  0.40,  0.60,  0.20],  # neuron 3
            [0.40,  0.30,  0.20,  0.60],  # neuron 4
        ]
        self._b2 = [-0.40, -0.35, -0.30, -0.25]

        # ---- MLP Weights: Hidden Layer 2 (4) -> Output (4 classes) ----
        self._w3 = [
            [0.10,  0.20,  0.30,  0.80],  # Low
            [0.20,  0.30,  0.70,  0.20],  # Normal
            [0.30,  0.70,  0.20,  0.10],  # High
            [0.80,  0.20,  0.10,  0.05],  # Critical
        ]
        self._b3 = [-0.50, -0.40, -0.30, -0.10]

   
    # Activation Functions
    

    def _sigmoid(self, x):
        """
        Function: _sigmoid
        Description:
            Computes the sigmoid activation: 1 / (1 + e^(-x)).
            Used in the binary classifier output layer.

        Parameters:
            x (float): Input value.

        Returns:
            float: Output in range (0, 1).
        
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _relu(self, x):
        """
        
        Function: _relu
        Description:
            Computes the ReLU activation: max(0, x).
            Used in hidden layers of the MLP.

        Parameters:
            x (float): Input value.

        Returns:
            float: Output >= 0.
        
        """
        return max(0.0, x)

    def _softmax(self, values):
        """
       
        Function: _softmax
        Description:
            Computes the softmax of a list of values, returning a
            probability distribution over output classes.

        Parameters:
            values (list[float]): Raw scores for each output class.

        Returns:
            list[float]: Probabilities summing to 1.0.
        
        """
        try:
            max_val = max(values)  # Numerical stability: subtract max
            exps    = [math.exp(v - max_val) for v in values]
            total   = sum(exps)
            if total == 0:
                return [1.0 / len(values)] * len(values)
            return [e / total for e in exps]
        except Exception:
            return [1.0 / len(values)] * len(values)

    
    # Forward Pass Helpers
    

    def _layer_forward(self, inputs, weights, biases, activation):
        """
        
        Function: _layer_forward
        Description:
            Performs a single layer forward pass: computes weighted sum
            for each neuron and applies the activation function.

        Parameters:
            inputs     (list[float]): Input values to this layer.
            weights    (list[list[float]]): Weight matrix (neurons x inputs).
            biases     (list[float]): Bias values per neuron.
            activation (callable): Activation function to apply.

        Returns:
            list[float]: Output values after activation.
    
        """
        outputs = []
        for neuron_idx, neuron_weights in enumerate(weights):
            # Weighted sum
            weighted_sum = sum(
                w * x for w, x in zip(neuron_weights, inputs)
            ) + biases[neuron_idx]
            outputs.append(activation(weighted_sum))
        return outputs

   
    # Main Prediction Function
    

    def predict(self, processed_request):
        """
        
        Function: predict
        Description:
            Runs both the binary urgency classifier and the MLP
            multi-class priority classifier on the feature vector
            from the preprocessed request.

        Parameters:
            processed_request (dict): Validated request containing
                                      'ann_feature_vector'.

        Returns:
            dict: {
                'priority_level'  : str  (Low/Normal/High/Critical),
                'priority_index'  : int  (0-3),
                'urgency_score'   : float (binary sigmoid output),
                'is_urgent'       : bool,
                'class_probs'     : list[float] (softmax probabilities),
                'feature_vector'  : list[float]
            }
        
        """
        feature_vector = processed_request.get("ann_feature_vector", [])

        if not feature_vector or len(feature_vector) != 6:
            raise ValueError(
                "ANNPriorityModule: 'ann_feature_vector' must be a list of 6 floats. "
                f"Received: {feature_vector}"
            )

        # --- Binary Classifier Forward Pass ---
        binary_raw = sum(
            w * x for w, x in zip(self._binary_weights, feature_vector)
        ) + self._binary_bias
        urgency_score = self._sigmoid(binary_raw)
        is_urgent     = urgency_score >= self.BINARY_URGENCY_THRESHOLD

        # --- MLP Forward Pass ---
        # Hidden Layer 1 (ReLU)
        hidden1 = self._layer_forward(feature_vector, self._w1, self._b1, self._relu)

        # Hidden Layer 2 (ReLU)
        hidden2 = self._layer_forward(hidden1, self._w2, self._b2, self._relu)

        # Output Layer (Softmax)
        output_raw   = self._layer_forward(hidden2, self._w3, self._b3, lambda x: x)
        class_probs  = self._softmax(output_raw)

        # Determine final priority class (highest probability)
        priority_index = class_probs.index(max(class_probs))

        # Override: if binary classifier says urgent + vehicle is emergency type
        is_emergency = processed_request.get("is_emergency_vehicle", False)
        severity     = processed_request.get("incident_severity", "Low")

        if is_emergency and severity == "Critical" and priority_index < 3:
            priority_index = 3  # Escalate to Critical
        elif is_emergency and severity == "High" and priority_index < 2:
            priority_index = 2  # Escalate to High

        priority_level = self.PRIORITY_LEVELS[priority_index]

        return {
            "priority_level" : priority_level,
            "priority_index" : priority_index,
            "urgency_score"  : round(urgency_score, 4),
            "is_urgent"      : is_urgent,
            "class_probs"    : [round(p, 4) for p in class_probs],
            "feature_vector" : feature_vector
        }

    def describe(self):
        """
        Function: describe
        Description:
            Returns a human-readable summary of the ANN architecture
            for display or debugging purposes.

        Returns:
            str: Multi-line architecture description.
        """
        return (
            "ANN Priority Module Architecture:\n"
            "  Binary Classifier  : 6 inputs -> Sigmoid -> Urgent / Not Urgent\n"
            "  MLP Classifier     : 6 inputs\n"
            "                        -> Hidden Layer 1 (4 neurons, ReLU)\n"
            "                        -> Hidden Layer 2 (4 neurons, ReLU)\n"
            "                        -> Output (4 classes, Softmax)\n"
            f"  Priority Classes   : {self.PRIORITY_LEVELS}\n"
            f"  Urgency Threshold  : {self.BINARY_URGENCY_THRESHOLD}"
        )
