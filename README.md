# SmartCity-Traffic-AI
Smart City Traffic & Emergency Response AI System
A comprehensive, multi-module AI application designed to manage urban traffic flow and coordinate emergency vehicle responses in a simulated city environment.

📋 Project Overview
The system processes structured requests from diverse road users—including ambulances, fire trucks, police, buses, and civilian vehicles—through an intelligent, modular pipeline to generate routing decisions, traffic signal control plans, and policy authorizations.

🧠 Core AI Techniques
This project integrates five distinct AI techniques into a cohesive pipeline:

ANN (Artificial Neural Networks): Multi-Layer Perceptron (MLP) for urgency prediction and priority classification.

Rule-Based Reasoning: First-Order Logic (FOL) Knowledge Base with forward-chaining rules for policy enforcement.

Constraint Satisfaction (CSP): Backtracking search algorithm for dynamic traffic signal phase allocation.

Graph Search Algorithms: BFS, UCS, and A* for optimal pathfinding.

Request Routing: A controller that selectively invokes pipeline modules based on request type for maximum efficiency.

🏗️ System Architecture
The system is structured into 7 independent modules, orchestrated by a central pipeline:

Module	Role
Preprocessor	Data normalization and ANN feature encoding.
Router	Maps request categories to the required execution pipeline.
ANN Priority	Predicts urgency (Low/Normal/High/Critical) via softmax classification.
Logic KB	Enforces policy and derives authorizations via forward chaining.
CSP Scheduler	Allocates signal phases across 5 intersections using backtracking.
Search/Nav	Calculates optimal routes using BFS, UCS, or A*.
Final Response	Aggregates all module outputs into human-readable results.
🗺️ City Topology
The simulation utilizes a directed graph consisting of 13 nodes (intersections/landmarks) and 18 edges. Five nodes are designated as signal-controlled zones managed by the CSP module.

📊 Visualizations
The system provides immediate visual feedback using matplotlib:

City Map: Displays the graph topology and highlights the active route.

ANN Chart: Bar chart showing probability distributions over priority classes.

CSP Grid: Displays active signal phase allocations for all signal zones.

🚀 Getting Started
1. Clone the Repository
Clone the project to your local machine:

Bash
git clone https://github.com/usmanaliclouds-jpg/SmartCity-Traffic-AI.git
cd SmartCity-Traffic-AI
2. Prerequisites & Installation
Ensure you have Python 3.10 installed. Install the required dependencies using pip:

Bash
pip install matplotlib networkx
Note for Linux Users: This system uses TkAgg as the backend for rendering GUI windows. If you encounter rendering errors, please ensure the tkinter package is installed on your system:

Bash
sudo apt-get install python3-tk
3. Execution
Launch the system controller via the terminal:

Bash
python main.py
