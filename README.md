# 🏙️ Smart City Traffic & Emergency Response AI System

A full-featured AI-powered traffic management and emergency response simulation with a graphical interface built using **matplotlib**. The system integrates multiple AI modules — including an Artificial Neural Network, a Knowledge Base, a Constraint Satisfaction Problem scheduler, and search-based navigation — into a single unified pipeline.

---

## 📸 Overview

The GUI provides **6 interactive tabs**:

| Tab | Description |
|-----|-------------|
| **Control Panel** | Submit requests via clickable dropdown-style rows |
| **City Map** | Live city road graph with highlighted routes |
| **ANN Module** | Neural network architecture and priority probability bars |
| **CSP Signals** | Traffic signal phase assignment per intersection zone |
| **Pipeline Flow** | Visual flow showing which modules ran and their status |
| **Response Panel** | Full formatted final response from the AI pipeline |

---

## 🗂️ Project Structure

```
smart-city-ai/
│
├── main.py                  # Entry point — launches the matplotlib GUI
│
└── modules/
    ├── __init__.py
    ├── preprocessor.py      # Module 1 — Input validation & normalization
    ├── router.py            # Module 2 — Pipeline routing by request category
    ├── ann_priority.py      # Module 3 — ANN-based urgency/priority prediction
    ├── logic_kb.py          # Module 4 — Logic rules & knowledge base validation
    ├── csp_scheduler.py     # Module 5 — CSP-based signal/control allocation
    ├── search_navigation.py # Module 6 — BFS / UCS / A* route finding
    └── final_response.py    # Module 7 — Aggregated final response generator
```

---

## ⚙️ Modules

### Module 1 — `preprocessor.py` — Input Preprocessor
Validates and normalizes all incoming request fields. Builds the feature vector used by the ANN module. Rejects invalid requests with descriptive error messages.

### Module 2 — `router.py` — Request Router
Determines which AI modules to run based on the request category:

| Category | Pipeline |
|----------|----------|
| `Route_Request` | Search |
| `Policy_Check` | Logic_KB |
| `Control_Allocation_Request` | Logic_KB → CSP |
| `Emergency_Response_Request` | ANN → Logic_KB → CSP → Search |
| `Integrated_City_Service_Request` | ANN → Logic_KB → CSP → Search → Final_Response |

### Module 3 — `ann_priority.py` — ANN Priority Module
A multi-layer perceptron (MLP) that classifies request urgency into **Low / Normal / High / Critical** using 6 input features (vehicle type, severity, time sensitivity, traffic density, priority claim, distance). Also runs a binary urgency classifier.

### Module 4 — `logic_kb.py` — Logic / Knowledge Base
Rule-based policy engine. Checks authorization, emergency corridor eligibility, signal override rights, and validates against the system knowledge base before any control action is executed.

### Module 5 — `csp_scheduler.py` — CSP Scheduler
Solves the traffic signal assignment as a Constraint Satisfaction Problem. Assigns Phase_A / Phase_B / Phase_C to each intersection while respecting conflict, timing, and emergency priority constraints.

### Module 6 — `search_navigation.py` — Search & Navigation
Finds optimal routes over the city road graph using:
- **BFS** — unweighted, fewest hops
- **UCS** — weighted, lowest cost
- **A\*** — weighted with Euclidean heuristic

### Module 7 — `final_response.py` — Final Response Layer
Aggregates outputs from all executed modules into a clean, selective response showing only the fields relevant to the current request type.

---

## 🛠️ Requirements

Make sure you have **Python 3.8+** installed. Then install the following dependencies:

```bash
pip install matplotlib numpy
```

> **Note:** No `tkinter` is required. The GUI uses matplotlib's built-in rendering. If you are running in a headless environment (no display), the system automatically falls back to the `Agg` backend.

### Full dependency list

| Package | Purpose |
|---------|---------|
| `matplotlib` | GUI rendering, all tabs and interactive widgets |
| `numpy` | Numerical computations for ANN and graph operations |

---

## 🚀 Installation & Setup

### Option 1 — Clone from GitHub

```bash
git clone https://github.com/usmanaliclouds-jpg/SmartCity-Traffic-AI.git
cd smart-city-ai
```

### Option 2 — Download ZIP

Click the green **Code** button on the repository page → **Download ZIP** → extract it.

---

### Install dependencies

```bash
pip install matplotlib numpy
```

---

### Run the application

```bash
python main.py
```

The GUI window will open automatically.

---

## 🖥️ How to Use

1. Open the **Control Panel** tab (default on launch).
2. **Click on any field row** to cycle through available options (e.g., change vehicle type, source, destination, severity).
3. Press the **▶ SUBMIT REQUEST** button at the bottom.
4. The system runs the AI pipeline and automatically switches to the **Response** tab.
5. Explore the other tabs to inspect:
   - The route highlighted on the **City Map**
   - The ANN priority prediction in **ANN Module**
   - The signal phase assignments in **CSP Signals**
   - Which modules executed in **Pipeline Flow**

---

## 🗺️ City Graph

The simulation covers **13 city nodes** connected by weighted road edges:

```
Police_HQ · Traffic_Control_Center · North_Station · River_Bridge
Stadium · East_Market · Central_Junction · West_Terminal
Airport_Road · South_Residential · City_Hospital
Industrial_Zone · Fire_Station
```

Signal-controlled intersections: `Central_Junction`, `North_Station`, `East_Market`, `River_Bridge`, `City_Hospital`

---

## 📋 Request Configuration Options

| Field | Options |
|-------|---------|
| **Category** | Route_Request, Policy_Check, Control_Allocation_Request, Emergency_Response_Request, Integrated_City_Service_Request |
| **Vehicle Type** | ambulance, fire_truck, police, civilian_car, bus |
| **Source / Destination** | Any of the 13 city nodes |
| **Severity** | Low, Medium, High, Critical |
| **Traffic Density** | Low, Medium, High |
| **Time Sensitive** | True, False |
| **Priority Claim** | True, False |
| **Control Zone** | Any signal-controlled intersection |

---

## 📦 Repository

```bash
git clone https://github.com/usmanaliclouds-jpg/SmartCity-Traffic-AI.git
```

> Replace `your-username` with your actual GitHub username before sharing.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is for educational and research purposes.
