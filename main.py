"""
=====================================================================
Smart City Traffic & Emergency Response AI System — GUI
=====================================================================
Description:
    A full graphical interface for the Smart City Traffic system.
    Built with matplotlib (no tkinter required). Provides:

        Tab 1 — Control Panel  : Submit requests via dropdown menus
        Tab 2 — City Map       : Live city road graph with route highlighted
        Tab 3 — ANN Diagram    : Neural network architecture + priority bars
        Tab 4 — CSP Signal Grid: Traffic signal phase assignment per zone
        Tab 5 — Pipeline Flow  : Which modules ran and their status
        Tab 6 — Response Panel : Full formatted final response

    All tabs update automatically after each request is submitted.

Usage:
    python gui.py
=====================================================================
"""

import sys
import os
import math
import uuid

# Add project root to path so modules/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import matplotlib.patheffects as pe
import numpy as np

from modules.preprocessor import InputPreprocessor
from modules.router import RequestRouter
from modules.ann_priority import ANNPriorityModule
from modules.logic_kb import LogicKnowledgeBase
from modules.csp_scheduler import CSPScheduler
from modules.search_navigation import SearchNavigationModule
from modules.final_response import FinalResponseLayer


# ─────────────────────────────────────────────────────────────────
# GLOBAL FONT SCALE  — change FS to resize all text at once
# ─────────────────────────────────────────────────────────────────
FS = 1.55   # scale factor applied to every fontsize

def fs(size):
    """Return scaled font size."""
    return round(size * FS, 1)

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE  — Mid-tone slate blue-grey (not too dark, not too light)
# ─────────────────────────────────────────────────────────────────
C = {
    # Backgrounds — comfortable mid-tone slate layers
    "bg"        : "#1e2a3a",        # main bg: slate blue-grey
    "panel"     : "#243040",        # panel base
    "panel2"    : "#2c3a50",        # card surface
    "border"    : "#3a5878",        # clear border
    "border2"   : "#2a4060",        # dimmer border variant

    # Primary accents — electric cyan/blue
    "accent"    : "#00c8f0",        # electric cyan (primary)
    "accent2"   : "#2090d8",        # royal blue (secondary)
    "accent3"   : "#00e8b4",        # teal-green (hospital)

    # Semantic colours
    "green"     : "#00e88a",        # success / route
    "yellow"    : "#f0c040",        # warning / weight labels
    "orange"    : "#f07030",        # phase A / heat
    "red"       : "#f03050",        # danger / emergency

    # Text hierarchy
    "text"      : "#e8f4ff",        # primary text
    "subtext"   : "#7a9ab8",        # muted / secondary text
    "dim"       : "#3a5068",        # disabled / very dim

    # Map-specific
    "node"      : "#2a3e58",        # default node fill
    "node_hl"   : "#00c8f0",        # highlighted node border
    "edge"      : "#304a68",        # road edge
    "route"     : "#00e88a",        # route overlay
    "emergency" : "#f03050",        # emergency route
    "hospital"  : "#00ddb0",        # hospital node (teal/cyan)
    "signal"    : "#263c58",        # signal zone node fill

    # CSP phase colours — vivid traffic lights
    "phase_a"   : "#f06020",        # Phase A — amber/orange
    "phase_b"   : "#f0c030",        # Phase B — gold
    "phase_c"   : "#00e88a",        # Phase C — green

    # Start / end markers
    "start_node": "#f0c040",        # gold start
    "end_node"  : "#00c8f0",        # cyan destination
}

PRIORITY_COLORS = {
    "Low"      : "#207850",         # muted green
    "Normal"   : "#1a5090",         # muted blue
    "High"     : "#c08010",         # amber
    "Critical" : "#c02030",         # red
}

PRIORITY_GLOW = {
    "Low"      : "#00e88a",
    "Normal"   : "#00c8f0",
    "High"     : "#f0c040",
    "Critical" : "#f03050",
}


# ─────────────────────────────────────────────────────────────────
# CITY GRAPH LAYOUT  (fixed 2-D positions for drawing)
# ─────────────────────────────────────────────────────────────────
NODE_POS = {
    "Police_HQ"             : (1.0, 9.0),
    "Traffic_Control_Center": (5.5, 9.5),
    "North_Station"         : (5.5, 7.5),
    "River_Bridge"          : (3.0, 8.0),
    "Stadium"               : (1.5, 6.0),
    "East_Market"           : (4.5, 5.5),
    "Central_Junction"      : (6.5, 5.5),
    "West_Terminal"         : (8.5, 5.0),
    "Airport_Road"          : (2.5, 3.5),
    "South_Residential"     : (5.0, 3.5),
    "City_Hospital"         : (4.0, 1.5),
    "Industrial_Zone"       : (9.5, 3.5),
    "Fire_Station"          : (9.5, 5.5),
}

EDGES_WEIGHTED = [
    ("Police_HQ",              "Traffic_Control_Center", 2),
    ("Police_HQ",              "River_Bridge",           4),
    ("Traffic_Control_Center", "North_Station",          4),
    ("Traffic_Control_Center", "Airport_Road",           6),
    ("North_Station",          "River_Bridge",           4),
    ("North_Station",          "Central_Junction",       3),
    ("North_Station",          "South_Residential",      3),
    ("River_Bridge",           "Stadium",                5),
    ("Stadium",                "East_Market",            3),
    ("Stadium",                "South_Residential",      5),
    ("East_Market",            "Central_Junction",       3),
    ("East_Market",            "City_Hospital",          3),
    ("Central_Junction",       "West_Terminal",          3),
    ("Central_Junction",       "South_Residential",      4),
    ("West_Terminal",          "Fire_Station",           2),
    ("West_Terminal",          "Industrial_Zone",        4),
    ("Airport_Road",           "South_Residential",      5),
    ("Airport_Road",           "City_Hospital",          3),
]

SIGNAL_ZONES = {"Central_Junction", "North_Station", "East_Market",
                "River_Bridge", "City_Hospital"}

VALID_LOCATIONS = list(NODE_POS.keys())
VALID_ZONES     = list(SIGNAL_ZONES)

VEHICLE_TYPES = ["ambulance", "fire_truck", "police", "civilian_car", "bus"]
CATEGORIES    = [
    "Route_Request",
    "Policy_Check",
    "Control_Allocation_Request",
    "Emergency_Response_Request",
    "Integrated_City_Service_Request",
]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
DENSITIES  = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────────────────────────
# HELPER — draw a rounded box with text
# ─────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, color, textcolor="#ffffff",
             fontsize=fs(8), alpha=0.92, radius=0.3):
    """
    ---------------------------------------------------------------
    Function: draw_box
    Description:
        Draws a rounded rectangle with centred label on the given
        matplotlib Axes. Used for node/module boxes throughout the
        GUI.

    Parameters:
        ax        : matplotlib Axes
        x, y      : bottom-left corner coordinates
        w, h      : width and height
        label     : text to display inside box
        color     : face colour string
        textcolor : label colour string
        fontsize  : label font size
        alpha     : box transparency
        radius    : corner rounding radius
    ---------------------------------------------------------------
    """
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=C["accent"], linewidth=1.4,
        alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=4,
            wrap=True)


# ─────────────────────────────────────────────────────────────────
# DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def draw_city_map(ax, route_path=None, is_emergency=False):
    """
    ---------------------------------------------------------------
    Function: draw_city_map
    Description:
        Renders the city road network as a graph on the given Axes.
        All edges are drawn first, then nodes, then labels.
        If route_path is given, the route edges are highlighted in
        green (or red for emergency). Edge weights are shown.

    Parameters:
        ax          : matplotlib Axes to draw on
        route_path  : list of node names forming the route, or None
        is_emergency: bool — uses red highlight if True
    ---------------------------------------------------------------
    """
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.5, 11.5)
    ax.axis("off")

    # Subtle grid lines for depth effect
    for gx in np.arange(0, 12, 1.5):
        ax.axvline(gx, color="#1a2636", linewidth=0.5, zorder=0, alpha=0.7)
    for gy in np.arange(0, 12, 1.5):
        ax.axhline(gy, color="#1a2636", linewidth=0.5, zorder=0, alpha=0.7)

    ax.set_title("  CITY ROAD NETWORK", color=C["accent"],
                 fontsize=fs(11), fontweight="bold", pad=10,
                 fontfamily="monospace", loc="left")

    route_set = set()
    if route_path and len(route_path) > 1:
        for i in range(len(route_path) - 1):
            route_set.add((route_path[i], route_path[i + 1]))
            route_set.add((route_path[i + 1], route_path[i]))

    # Draw edges — non-route first for proper layering
    for (n1, n2, w) in EDGES_WEIGHTED:
        x1, y1 = NODE_POS[n1]
        x2, y2 = NODE_POS[n2]
        on_route = (n1, n2) in route_set

        if on_route:
            route_col = C["emergency"] if is_emergency else C["route"]
            # Glow effect: wide translucent underlay + sharp line
            ax.plot([x1, x2], [y1, y2], color=route_col,
                    linewidth=7.0, alpha=0.18, zorder=1, solid_capstyle="round")
            ax.plot([x1, x2], [y1, y2], color=route_col,
                    linewidth=2.5, alpha=1.0, zorder=2, solid_capstyle="round")
        else:
            ax.plot([x1, x2], [y1, y2], color=C["edge"],
                    linewidth=0.9, linestyle="--", zorder=1, alpha=0.55)

        # Weight label on edges
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(w), fontsize=fs(6), color=C["yellow"],
                ha="center", va="center", zorder=3, alpha=0.75,
                bbox=dict(facecolor=C["bg"], edgecolor="none", pad=0.8, alpha=0.85))

    # Draw nodes
    for node, (x, y) in NODE_POS.items():
        on_route   = route_path and node in route_path
        is_signal  = node in SIGNAL_ZONES
        is_start   = route_path and node == route_path[0]
        is_end     = route_path and node == route_path[-1]
        is_hospital = node == "City_Hospital"

        # Choose colours and size per node role
        if is_start:
            fc, ec, r, lw = C["start_node"], "#ffffff", 0.40, 2.0
        elif is_end:
            fc  = C["emergency"] if is_emergency else C["end_node"]
            ec  = "#ffffff"
            r, lw = 0.40, 2.0
        elif is_hospital:
            fc, ec, r, lw = C["hospital"], "#00ffcc", 0.33, 2.0
        elif on_route:
            fc  = C["emergency"] if is_emergency else C["route"]
            ec  = "#ffffff"
            r, lw = 0.32, 1.8
        elif is_signal:
            fc, ec, r, lw = C["signal"], C["yellow"], 0.30, 1.5
        else:
            fc, ec, r, lw = C["node"], C["accent2"], 0.27, 1.2

        # Glow halo for highlighted nodes
        if is_start or is_end or is_hospital or on_route:
            halo_col = fc
            halo = Circle((x, y), r + 0.14, facecolor=halo_col,
                          edgecolor="none", alpha=0.15, zorder=3)
            ax.add_patch(halo)

        circ = Circle((x, y), r, facecolor=fc, edgecolor=ec,
                      linewidth=lw, zorder=4)
        ax.add_patch(circ)

        # Icon-like short label inside node
        if is_hospital:
            ax.text(x, y, "✚", ha="center", va="center",
                    fontsize=fs(8), color="#ffffff", fontweight="bold", zorder=5)
        elif is_start and route_path:
            ax.text(x, y, "S", ha="center", va="center",
                    fontsize=fs(7), color=C["bg"], fontweight="bold", zorder=5)
        elif is_end and route_path:
            ax.text(x, y, "D", ha="center", va="center",
                    fontsize=fs(7), color=C["bg"], fontweight="bold", zorder=5)

        # Node name label below
        short = node.replace("_", "\n")
        label_col = C["hospital"] if is_hospital else (C["text"] if on_route or is_signal else C["subtext"])
        ax.text(x, y - r - 0.2, short, ha="center", va="top",
                fontsize=fs(5.2), color=label_col, zorder=5,
                fontfamily="monospace", linespacing=1.1)


def draw_ann_diagram(ax, ann_result=None):
    """
    ---------------------------------------------------------------
    Function: draw_ann_diagram
    Description:
        Draws the MLP architecture diagram (input → H1 → H2 → output)
        with activation labels. If ann_result is provided, highlights
        the winning output class and shows the probability bar chart
        beside the network.

    Parameters:
        ax         : matplotlib Axes
        ann_result : dict from ANNPriorityModule.predict(), or None
    ---------------------------------------------------------------
    """
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Subtle background gradient bands
    for i, bx in enumerate(np.linspace(0, 10, 6)):
        ax.axvline(bx, color="#1a2636", linewidth=12, alpha=0.3, zorder=0)

    ax.set_title("  ANN PRIORITY MODULE", color=C["accent"],
                 fontsize=fs(11), fontweight="bold", pad=10,
                 fontfamily="monospace", loc="left")

    layers = {
        "Input"  : (1.2,  ["Vehicle", "Severity", "TimeSens", "Density", "Priority", "Distance"]),
        "H1"     : (3.5,  ["N1", "N2", "N3", "N4"]),
        "H2"     : (5.8,  ["N1", "N2", "N3", "N4"]),
        "Output" : (8.1,  ["Low", "Normal", "High", "Critical"]),
    }

    layer_y = {
        6: [1.5, 2.8, 4.1, 5.4, 6.7, 8.0],
        4: [2.2, 3.8, 5.4, 7.0],
    }

    node_coords = {}
    for lname, (lx, lnodes) in layers.items():
        ys = layer_y[len(lnodes)]
        for i, nname in enumerate(lnodes):
            y   = ys[i]
            key = (lname, nname)
            node_coords[key] = (lx, y)

            if lname == "Output":
                base_col = PRIORITY_COLORS.get(nname, C["node"])
                glow_col = PRIORITY_GLOW.get(nname, C["accent"])
                is_winner = ann_result and ann_result["priority_level"] == nname
                alpha = 1.0 if is_winner else 0.28
                # Winner glow halo
                if is_winner:
                    halo = Circle((lx, y), 0.42, facecolor=glow_col,
                                  edgecolor="none", alpha=0.22, zorder=2)
                    ax.add_patch(halo)
                circ = Circle((lx, y), 0.30, facecolor=base_col,
                              edgecolor=glow_col if is_winner else C["border"],
                              linewidth=2.0 if is_winner else 0.8,
                              alpha=alpha, zorder=3)
            else:
                # Hidden / input nodes — glassy look
                col   = C["panel2"]
                ec    = C["accent"] if lname == "Input" else C["accent2"]
                circ  = Circle((lx, y), 0.28, facecolor=col,
                               edgecolor=ec, linewidth=1.0, alpha=0.9, zorder=3)

            ax.add_patch(circ)
            ax.text(lx, y, nname[:3], ha="center", va="center",
                    fontsize=fs(4.8), color=C["text"], fontweight="bold", zorder=4)

    # Connections between layers
    def connect_layers(l1_name, l2_name):
        l1_nodes = layers[l1_name][1]
        l2_nodes = layers[l2_name][1]
        ys1 = layer_y[len(l1_nodes)]
        ys2 = layer_y[len(l2_nodes)]
        x1  = layers[l1_name][0]
        x2  = layers[l2_name][0]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + 0.28, x2 - 0.28], [y1, y2],
                        color=C["accent2"], linewidth=0.35, alpha=0.28, zorder=1)

    connect_layers("Input", "H1")
    connect_layers("H1",    "H2")
    connect_layers("H2",    "Output")

    # Layer header labels + activation tags
    for lname, (lx, _) in layers.items():
        activation = {"Input": "Features", "H1": "ReLU", "H2": "ReLU", "Output": "Softmax"}

        # Header box
        hbox = FancyBboxPatch((lx - 0.55, 8.65), 1.1, 0.38,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor=C["panel2"], edgecolor=C["accent"],
                              linewidth=0.8, alpha=0.9, zorder=3)
        ax.add_patch(hbox)
        ax.text(lx, 8.84, lname, ha="center", va="center",
                fontsize=fs(7.5), color=C["accent"], fontweight="bold",
                fontfamily="monospace", zorder=4)
        ax.text(lx, 8.35, activation[lname], ha="center", va="center",
                fontsize=fs(6), color=C["subtext"], fontfamily="monospace")

    # Probability bar chart (right side)
    if ann_result:
        probs  = ann_result.get("class_probs", [0.25, 0.25, 0.25, 0.25])
        labels = ["Low", "Normal", "High", "Critical"]
        bar_x  = 9.15
        bar_w  = 0.48
        bar_ys = [2.2, 3.8, 5.4, 7.0]

        for i, (label, prob) in enumerate(zip(labels, probs)):
            bh   = prob * 1.7
            col  = PRIORITY_GLOW.get(label, C["accent"])
            is_w = ann_result["priority_level"] == label
            alpha = 1.0 if is_w else 0.30

            # Bar glow
            if is_w:
                glow_rect = Rectangle((bar_x - 0.04, bar_ys[i] - bh / 2 - 0.04),
                                      bar_w + 0.08, bh + 0.08,
                                      facecolor=col, edgecolor="none",
                                      alpha=0.18, zorder=2)
                ax.add_patch(glow_rect)

            rect = Rectangle((bar_x, bar_ys[i] - bh / 2), bar_w, bh,
                              facecolor=PRIORITY_COLORS[label], edgecolor=col if is_w else "none",
                              linewidth=1.0, alpha=alpha, zorder=3)
            ax.add_patch(rect)
            ax.text(bar_x + bar_w / 2, bar_ys[i] + bh / 2 + 0.15,
                    f"{prob:.2f}", ha="center", va="bottom",
                    fontsize=fs(6), color=col if is_w else C["subtext"], zorder=4,
                    fontweight="bold" if is_w else "normal")

        ax.text(bar_x + bar_w / 2, 9.0, "Prob.", ha="center", fontsize=fs(6.5),
                color=C["accent"], fontfamily="monospace", fontweight="bold")

        # Status bar at the bottom
        score     = ann_result.get("urgency_score", 0)
        is_urgent = ann_result["is_urgent"]
        prio      = ann_result["priority_level"]
        bar_col   = PRIORITY_GLOW.get(prio, C["accent"])

        status_box = FancyBboxPatch((0.4, 0.08), 9.2, 0.55,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=C["panel2"], edgecolor=bar_col,
                                    linewidth=1.2, zorder=3)
        ax.add_patch(status_box)
        ax.text(5.0, 0.38,
                f"Binary Urgency Score: {score:.4f}   |   "
                f"Urgent: {'YES' if is_urgent else 'NO'}   |   "
                f"Priority: {prio}",
                ha="center", va="center", fontsize=fs(7.5),
                color=bar_col if is_urgent else C["subtext"],
                fontfamily="monospace", fontweight="bold", zorder=4)
    else:
        ax.text(5.0, 0.4, "Submit a request to see ANN output",
                ha="center", va="center", fontsize=fs(8),
                color=C["subtext"], fontfamily="monospace")


def draw_csp_grid(ax, csp_result=None):
    """
    ---------------------------------------------------------------
    Function: draw_csp_grid
    Description:
        Renders the CSP signal assignment as a grid of traffic
        light indicators — one column per intersection, coloured by
        the assigned phase (Phase_A=orange, Phase_B=yellow,
        Phase_C=green). Shows constraint edges between nodes.

    Parameters:
        ax         : matplotlib Axes
        csp_result : dict from CSPScheduler.allocate(), or None
    ---------------------------------------------------------------
    """
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Subtle background pattern
    for gy in np.arange(0, 8.5, 0.8):
        ax.axhline(gy, color="#1a2636", linewidth=0.6, alpha=0.5, zorder=0)

    ax.set_title("  CSP SIGNAL ALLOCATION", color=C["accent"],
                 fontsize=fs(11), fontweight="bold", pad=10,
                 fontfamily="monospace", loc="left")

    zones       = ["Central_Junction", "North_Station", "East_Market",
                   "River_Bridge", "City_Hospital"]
    phase_color = {"Phase_A": C["phase_a"], "Phase_B": C["phase_b"],
                   "Phase_C": C["phase_c"]}
    phase_glow  = {"Phase_A": "#ff8040", "Phase_B": "#ffe060", "Phase_C": "#00ff99"}

    assignment = {}
    if csp_result and csp_result.get("signal_sequence"):
        assignment = csp_result["signal_sequence"]

    xs = [1.5, 3.3, 5.1, 6.9, 8.7]

    # Constraint arcs
    constraints = [(0, 2, "conflict"), (1, 3, "coordination")]
    for (i, j, label) in constraints:
        x1, x2 = xs[i] + 0.6, xs[j] + 0.6
        arc_col = C["red"] if label == "conflict" else C["accent2"]
        ax.annotate("", xy=(x2, 6.7), xytext=(x1, 6.7),
                    arrowprops=dict(arrowstyle="<->", color=arc_col,
                                   lw=1.4, connectionstyle="arc3,rad=0.28"))
        ax.text((x1 + x2) / 2, 7.2, label, ha="center", fontsize=fs(6.5),
                color=arc_col, fontfamily="monospace", fontweight="bold")

    # Emergency priority annotation
    ax.annotate("", xy=(xs[4] + 0.6, 6.6), xytext=(xs[4] + 0.6, 5.6),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.8))
    ax.text(xs[4] + 0.6, 7.15, "emerg.\npriority", ha="center",
            fontsize=fs(6), color=C["green"], fontfamily="monospace", fontweight="bold")

    for idx, (zone, x) in enumerate(zip(zones, xs)):
        phase   = assignment.get(zone, None)
        fc      = phase_color.get(phase, C["dim"]) if phase else C["dim"]
        gl      = phase_glow.get(phase, C["border"]) if phase else C["border"]
        is_ctrl = csp_result and csp_result.get("control_zone") == zone if csp_result else False
        is_hosp = zone == "City_Hospital"

        # Outer card glow for active/controlled zone
        if is_ctrl or is_hosp:
            glow_box = FancyBboxPatch((x - 0.08, 0.88), 1.36, 5.24,
                                     boxstyle="round,pad=0.1,rounding_size=0.25",
                                     facecolor="none",
                                     edgecolor=C["hospital"] if is_hosp else C["accent"],
                                     linewidth=2.5, alpha=0.6, zorder=1)
            ax.add_patch(glow_box)

        # Zone card background
        box = FancyBboxPatch((x, 1.0), 1.2, 5.0,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=C["panel2"],
                             edgecolor=C["accent"] if is_ctrl else (C["hospital"] if is_hosp else C["border"]),
                             linewidth=1.8 if (is_ctrl or is_hosp) else 0.8, zorder=2)
        ax.add_patch(box)

        # Traffic light housing
        housing = FancyBboxPatch((x + 0.28, 2.0), 0.64, 3.1,
                                 boxstyle="round,pad=0.05,rounding_size=0.12",
                                 facecolor="#243040", edgecolor="#3a5068",
                                 linewidth=0.8, zorder=3)
        ax.add_patch(housing)

        light_colors_all = [C["phase_a"], C["phase_b"], C["phase_c"]]
        light_glows      = ["#ff8040", "#ffe060", "#00ff99"]
        light_labels     = ["A", "B", "C"]
        light_ys         = [4.35, 3.50, 2.70]

        for li, (ly, lc_full, lg, ll) in enumerate(zip(light_ys, light_colors_all, light_glows, light_labels)):
            active = (phase == f"Phase_{ll}")
            color  = lc_full if active else "#243040"
            ec_col = lg if active else "#3a5068"

            # Glow halo for active light
            if active:
                halo = Circle((x + 0.60, ly), 0.28, facecolor=lc_full,
                              edgecolor="none", alpha=0.25, zorder=3)
                ax.add_patch(halo)

            circ = Circle((x + 0.60, ly), 0.21,
                          facecolor=color, edgecolor=ec_col,
                          linewidth=1.8 if active else 0.5,
                          alpha=1.0 if active else 0.35, zorder=4)
            ax.add_patch(circ)

        # Phase label badge
        phase_lbl = phase if phase else "—"
        badge_col = fc if phase else C["dim"]
        badge = FancyBboxPatch((x + 0.15, 1.32), 0.90, 0.38,
                               boxstyle="round,pad=0.04,rounding_size=0.1",
                               facecolor=C["bg"], edgecolor=badge_col,
                               linewidth=0.8, zorder=4)
        ax.add_patch(badge)
        ax.text(x + 0.60, 1.52, phase_lbl, ha="center", va="center",
                fontsize=fs(6), color=badge_col, fontweight="bold",
                fontfamily="monospace", zorder=5)

        # Zone name label
        short   = zone.replace("_", "\n")
        lbl_col = C["hospital"] if is_hosp else C["text"]
        ax.text(x + 0.60, 0.62, short, ha="center", va="center",
                fontsize=fs(5.2), color=lbl_col, fontfamily="monospace",
                zorder=5, linespacing=1.1)

        if is_ctrl:
            ax.text(x + 0.60, 5.35, "★ CTRL", ha="center", fontsize=fs(6),
                    color=C["yellow"], fontweight="bold", zorder=5,
                    fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(color=C["phase_a"], label="Phase A — Amber"),
        mpatches.Patch(color=C["phase_b"], label="Phase B — Gold"),
        mpatches.Patch(color=C["phase_c"], label="Phase C — Emergency"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=fs(6.5),
              facecolor=C["panel2"], edgecolor=C["border"],
              labelcolor=C["text"], framealpha=0.9,
              borderpad=0.8, labelspacing=0.6)

    if csp_result:
        status = "FEASIBLE ✓" if csp_result["feasible"] else "INFEASIBLE ✗"
        col    = C["green"] if csp_result["feasible"] else C["red"]
        status_box = FancyBboxPatch((1.8, -0.05), 8.4, 0.52,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor=C["panel2"], edgecolor=col,
                                    linewidth=1.2, zorder=3)
        ax.add_patch(status_box)
        ax.text(6.0, 0.22,
                f"CSP Status: {status}   |   Action: {csp_result['assigned_action']}",
                ha="center", fontsize=fs(6.5), color=col, fontfamily="monospace",
                fontweight="bold", zorder=4)


def draw_pipeline(ax, pipeline=None, results=None):
    """
    ---------------------------------------------------------------
    Function: draw_pipeline
    Description:
        Draws the processing pipeline as a horizontal flow diagram.
        Each module is a box; active modules are bright, skipped ones
        are dim. Arrows connect them in sequence. Status badges show
        APPROVED / REJECTED / FOUND / SKIPPED per module.

    Parameters:
        ax       : matplotlib Axes
        pipeline : list of module name strings, or None
        results  : dict of pipeline results, or None
    ---------------------------------------------------------------
    """
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Background depth lines
    for gy in np.arange(0, 5.5, 0.8):
        ax.axhline(gy, color="#1a2636", linewidth=0.7, alpha=0.5, zorder=0)

    ax.set_title("  PIPELINE FLOW", color=C["accent"],
                 fontsize=fs(11), fontweight="bold", pad=10,
                 fontfamily="monospace", loc="left")

    all_modules = [
        ("Preprocess", "preprocessor"),
        ("Router",     "router"),
        ("ANN",        "ann"),
        ("Logic/KB",   "logic"),
        ("CSP",        "csp"),
        ("Search",     "search"),
        ("Response",   "response"),
    ]

    active_names = set(pipeline) if pipeline else set()
    active_names.update({"Preprocess", "Router", "Response", "Final_Response"})

    statuses = {}
    if results:
        if results.get("ann_result"):
            statuses["ANN"] = results["ann_result"]["priority_level"]
        if results.get("logic_result"):
            pol = results["logic_result"]["policy_status"]
            statuses["Logic/KB"] = pol
        if results.get("csp_result"):
            statuses["CSP"] = "FEASIBLE" if results["csp_result"]["feasible"] else "INFEASIBLE"
        if results.get("search_result"):
            sr = results["search_result"]
            statuses["Search"] = sr["algorithm"] if sr["found"] else "NO ROUTE"

    box_w = 1.52
    box_h = 1.05
    gap   = 0.46
    total = len(all_modules)
    start_x = (14 - total * (box_w + gap) + gap) / 2

    for i, (label, key) in enumerate(all_modules):
        x      = start_x + i * (box_w + gap)
        y      = 2.0
        active = (label in active_names
                  or label in (pipeline or [])
                  or key in str(pipeline or "").lower())

        # Module icon characters
        icons = {
            "Preprocess": "⚙", "Router": "⇄", "ANN": "◉",
            "Logic/KB": "⊢", "CSP": "⊗", "Search": "⊘", "Response": "✓"
        }
        icon = icons.get(label, "▪")

        fc     = C["panel2"] if active else C["panel"]
        ec     = C["accent"] if active else C["border2"]
        alpha  = 1.0 if active else 0.4

        # Glow box for active modules
        if active:
            glow = FancyBboxPatch((x - 0.06, y - 0.06), box_w + 0.12, box_h + 0.12,
                                  boxstyle="round,pad=0.08,rounding_size=0.18",
                                  facecolor="none", edgecolor=C["accent"],
                                  linewidth=0.6, alpha=0.35, zorder=2)
            ax.add_patch(glow)

        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.08,rounding_size=0.18",
                             facecolor=fc, edgecolor=ec,
                             linewidth=1.8 if active else 0.7,
                             alpha=alpha, zorder=3)
        ax.add_patch(box)

        ax.text(x + box_w / 2, y + box_h * 0.65, icon,
                ha="center", va="center", fontsize=fs(11),
                color=C["accent"] if active else C["dim"],
                zorder=4)
        ax.text(x + box_w / 2, y + box_h * 0.25, label,
                ha="center", va="center", fontsize=fs(6.8),
                color=C["text"] if active else C["subtext"],
                fontweight="bold", fontfamily="monospace", zorder=4)

        # Status badge below box
        status = statuses.get(label, "")
        if status:
            scol = (C["green"]  if status in ("APPROVED", "FEASIBLE", "A*", "BFS", "UCS", "Low", "Normal")
                    else C["yellow"] if status in ("High",)
                    else C["red"]    if status in ("REJECTED", "INFEASIBLE", "NO ROUTE", "Critical")
                    else C["subtext"])
            badge = FancyBboxPatch((x + 0.1, y - 0.52), box_w - 0.2, 0.36,
                                   boxstyle="round,pad=0.04,rounding_size=0.08",
                                   facecolor=C["bg"], edgecolor=scol,
                                   linewidth=0.8, zorder=3)
            ax.add_patch(badge)
            ax.text(x + box_w / 2, y - 0.34, status,
                    ha="center", va="center", fontsize=fs(5.5),
                    color=scol, fontfamily="monospace", fontweight="bold", zorder=4)

        # Arrow to next module
        if i < total - 1:
            arrow_col = C["accent"] if active else C["dim"]
            ax.annotate("",
                        xy=(x + box_w + gap, y + box_h / 2),
                        xytext=(x + box_w, y + box_h / 2),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=arrow_col,
                            lw=1.6 if active else 0.5
                        ), zorder=2)

    # Active pipeline label
    if pipeline:
        pipe_str = " → ".join(pipeline)
        pipe_box = FancyBboxPatch((1.5, 0.28), 11.0, 0.48,
                                  boxstyle="round,pad=0.05,rounding_size=0.12",
                                  facecolor=C["panel2"], edgecolor=C["border"],
                                  linewidth=0.8, zorder=3)
        ax.add_patch(pipe_box)
        ax.text(7.0, 0.54, f"Active Pipeline:  {pipe_str}",
                ha="center", fontsize=fs(6.8), color=C["accent"],
                fontfamily="monospace", fontweight="bold", zorder=4)


def draw_response_panel(ax, final_response=None):
    """
    ---------------------------------------------------------------
    Function: draw_response_panel
    Description:
        Renders the final response as a formatted text dashboard
        on the given Axes. Shows request ID, status, and all
        component outputs in a clean monospaced layout.

    Parameters:
        ax             : matplotlib Axes
        final_response : dict from FinalResponseLayer.generate(), or None
    ---------------------------------------------------------------
    """
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Vertical accent stripe on the left
    stripe = Rectangle((0, 0), 0.18, 10, facecolor=C["accent"],
                        edgecolor="none", alpha=0.18, zorder=1)
    ax.add_patch(stripe)

    ax.set_title("  FINAL RESPONSE", color=C["accent"],
                 fontsize=fs(11), fontweight="bold", pad=10,
                 fontfamily="monospace", loc="left")

    if not final_response:
        ax.text(5, 5, "No request submitted yet.\nUse the Control Panel tab.",
                ha="center", va="center", fontsize=fs(10),
                color=C["subtext"], fontfamily="monospace")
        return

    status = final_response.get("status", "N/A")
    scol   = C["green"] if status == "SUCCESS" else C["red"]

    lines = [
        ("Request ID     :", final_response.get("request_id", "N/A"),        C["text"]),
        ("Category       :", final_response.get("request_category", "N/A"),  C["accent"]),
        ("Vehicle        :", final_response.get("vehicle_type", "N/A"),       C["text"]),
        ("From → To      :",
         f"{final_response.get('source','?')} → {final_response.get('destination','?')}",
         C["text"]),
        ("Status         :", status,                                           scol),
        ("", "", ""),
    ]

    comps = final_response.get("components", {})

    if "priority" in comps:
        p    = comps["priority"]
        pcol = PRIORITY_GLOW.get(p["priority_level"], C["text"])
        lines += [
            ("── ANN Priority ──", "", C["accent"]),
            ("Priority Level :", p["priority_level"],          pcol),
            ("Urgency Score  :", str(p["urgency_score"]),      C["text"]),
            ("Is Urgent      :", "YES" if p["is_urgent"] else "NO",
             C["green"] if p["is_urgent"] else C["subtext"]),
            ("", "", ""),
        ]

    if "policy" in comps:
        pol  = comps["policy"]
        pcol = C["green"] if pol["policy_status"] == "APPROVED" else C["red"]
        lines += [
            ("── Logic / KB  ──", "", C["accent"]),
            ("Policy Status  :", pol["policy_status"],   pcol),
            ("Authorized     :", "YES" if pol["authorized"] else "NO",
             C["green"] if pol["authorized"] else C["red"]),
            ("Emerg. Corridor:", "ACTIVE" if pol["emergency_corridor"] else "NO",
             C["green"] if pol["emergency_corridor"] else C["subtext"]),
            ("Signal Override:", "GRANTED" if pol["signal_override"] else "NO",
             C["green"] if pol["signal_override"] else C["subtext"]),
            ("", "", ""),
        ]

    if "signal_control" in comps:
        sc   = comps["signal_control"]
        fcol = C["green"] if sc["feasible"] else C["red"]
        lines += [
            ("── CSP Signals ──", "", C["accent"]),
            ("Control Action :", sc["assigned_action"][:40], fcol),
            ("Feasible       :", "YES" if sc["feasible"] else "NO", fcol),
            ("", "", ""),
        ]
        for zone, phase in sc.get("signal_sequence", {}).items():
            pc = {"Phase_A": C["phase_a"], "Phase_B": C["phase_b"],
                  "Phase_C": C["phase_c"]}.get(phase, C["text"])
            lines.append((f"  {zone:<22}", phase, pc))
        lines.append(("", "", ""))

    if "navigation" in comps:
        nav  = comps["navigation"]
        fcol = C["green"] if nav["route_found"] else C["red"]
        lines += [
            ("── Navigation  ──", "", C["accent"]),
            ("Algorithm      :", nav["algorithm"],   C["yellow"]),
            ("Route Found    :", "YES" if nav["route_found"] else "NO", fcol),
        ]
        if nav["route_found"]:
            path_str = " → ".join(nav["path"])
            if len(path_str) > 55:
                mid = len(nav["path"]) // 2
                lines.append(("Route          :", " → ".join(nav["path"][:mid + 1]), C["green"]))
                lines.append(("               ", " → ".join(nav["path"][mid:]),      C["green"]))
            else:
                lines.append(("Route          :", path_str, C["green"]))
            lines.append(("Hops           :", str(nav["hops"]),          C["text"]))
            if nav["total_cost"] is not None:
                lines.append(("Cost (units)   :", str(nav["total_cost"]), C["text"]))

    # Render lines
    y_pos = 9.5
    dy    = 0.43
    for (label, value, color) in lines:
        if label == "":
            y_pos -= dy * 0.4
            continue
        if label.startswith("──"):
            # Section divider with subtle underline
            ax.text(0.4, y_pos, label, fontsize=fs(7.5), color=color,
                    fontfamily="monospace", fontweight="bold", zorder=3)
            ax.plot([0.35, 9.65], [y_pos - 0.08, y_pos - 0.08],
                    color=color, linewidth=0.6, alpha=0.4, zorder=2)
        else:
            ax.text(0.4, y_pos, label, fontsize=fs(6.8),
                    color=C["subtext"], fontfamily="monospace", zorder=3)
            ax.text(4.1, y_pos, value, fontsize=fs(6.8),
                    color=color, fontfamily="monospace", fontweight="bold", zorder=3)
        y_pos -= dy
        if y_pos < 0.2:
            break


# ─────────────────────────────────────────────────────────────────
# MAIN GUI CLASS
# ─────────────────────────────────────────────────────────────────

class SmartCityGUI:
    """
    Main GUI controller. Manages the matplotlib figure, tab switching,
    dropdown widgets, and pipeline execution.
    """

    def __init__(self):
        """
        ---------------------------------------------------------------
        Function: __init__
        Description:
            Initialises all AI modules, sets up the matplotlib figure
            and all sub-axes, then draws the initial empty state of
            all panels.
        ---------------------------------------------------------------
        """
        # ── AI Modules ──
        self.preprocessor   = InputPreprocessor()
        self.router         = RequestRouter()
        self.ann            = ANNPriorityModule()
        self.logic_kb       = LogicKnowledgeBase()
        self.csp            = CSPScheduler()
        self.search         = SearchNavigationModule()
        self.response_layer = FinalResponseLayer()

        # ── State ──
        self.last_results  = None
        self.last_response = None
        self.current_tab   = 0

        # ── Selections ──
        self.sel_category = CATEGORIES[3]
        self.sel_vehicle  = VEHICLE_TYPES[0]
        self.sel_source   = "Central_Junction"
        self.sel_dest     = "City_Hospital"
        self.sel_severity = "Critical"
        self.sel_density  = "High"
        self.sel_zone     = "Central_Junction"
        self.sel_time     = True
        self.sel_priority = True

        # ── Build Figure ──
        plt.rcParams.update({
            "figure.facecolor" : C["bg"],
            "text.color"       : C["text"],
            "font.family"      : "monospace",
        })

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(C["bg"])
        self.fig.canvas.manager.set_window_title(
            "Smart City Traffic & Emergency Response AI System"
        )

        self._build_layout()
        self._draw_all()

        plt.tight_layout(rect=[0, 0.0, 1, 0.97])
        plt.show()

    # ── Layout ──────────────────────────────────────────────────

    def _build_layout(self):
        """
        ---------------------------------------------------------------
        Function: _build_layout
        Description:
            Constructs the figure layout using GridSpec. Creates all
            axes for the title bar, tab buttons, control panel, and
            the five visualisation panels (city map, ANN, CSP,
            pipeline, response).
        ---------------------------------------------------------------
        """
        gs = GridSpec(
            10, 12,
            figure=self.fig,
            hspace=0.05, wspace=0.3,
            left=0.02, right=0.98,
            top=0.96, bottom=0.04
        )

        # ── Title bar ──────────────────────────────────────────
        self.ax_title = self.fig.add_subplot(gs[0, :])
        self.ax_title.set_facecolor(C["panel"])
        self.ax_title.axis("off")

        # Left accent bar in title (drawn as a Rectangle to avoid axvline transform restriction)
        accent_bar = Rectangle((0, 0), 0.006, 1, facecolor=C["accent"],
                               edgecolor="none", alpha=0.9, zorder=5,
                               transform=self.ax_title.transAxes, clip_on=False)
        self.ax_title.add_patch(accent_bar)

        self.ax_title.text(
            0.5, 0.55,
            "◈  SMART CITY TRAFFIC & EMERGENCY RESPONSE AI SYSTEM  ◈",
            ha="center", va="center", fontsize=fs(13), fontweight="bold",
            color=C["accent"], fontfamily="monospace",
            transform=self.ax_title.transAxes
        )
        self.ax_title.text(
            0.5, 0.10,
            "Artificial Intelligence Pipeline  •  Real-time Traffic Management  •  Emergency Dispatch",
            ha="center", va="center", fontsize=fs(6.5),
            color=C["subtext"], fontfamily="monospace",
            transform=self.ax_title.transAxes
        )

        # ── Tab buttons ────────────────────────────────────────
        tab_labels = ["Control Panel", "City Map", "ANN Module",
                      "CSP Signals", "Pipeline", "Response"]
        self.tab_axes = []
        for i, label in enumerate(tab_labels):
            ax_tab = self.fig.add_subplot(gs[1, i * 2: i * 2 + 2])
            ax_tab.set_facecolor(C["accent"] if i == 0 else C["panel2"])
            ax_tab.axis("off")
            # Bottom indicator bar for active tab
            if i == 0:
                indicator = Rectangle((0, 0), 1, 0.06, facecolor=C["accent"],
                                      edgecolor="none", alpha=1.0, zorder=5,
                                      transform=ax_tab.transAxes, clip_on=False)
                ax_tab.add_patch(indicator)
            ax_tab.text(0.5, 0.5, label, ha="center", va="center",
                        fontsize=fs(7.5), fontweight="bold",
                        color=C["bg"] if i == 0 else C["subtext"],
                        fontfamily="monospace",
                        transform=ax_tab.transAxes)
            ax_tab.set_picker(True)
            self.tab_axes.append(ax_tab)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # ── Content axes ────────────────────────────────────────
        self.ax_control  = self.fig.add_subplot(gs[2:, :])
        self.ax_map      = self.fig.add_subplot(gs[2:, :])
        self.ax_ann      = self.fig.add_subplot(gs[2:, :])
        self.ax_csp      = self.fig.add_subplot(gs[2:, :])
        self.ax_pipeline = self.fig.add_subplot(gs[2:, :])
        self.ax_response = self.fig.add_subplot(gs[2:, :])

        self.content_axes = [
            self.ax_control, self.ax_map, self.ax_ann,
            self.ax_csp, self.ax_pipeline, self.ax_response
        ]

        # ── Submit button ───────────────────────────────────────
        self.ax_submit = self.fig.add_axes([0.42, 0.006, 0.16, 0.038])
        self.btn_submit = Button(
            self.ax_submit, "▶  SUBMIT REQUEST",
            color=C["accent"], hovercolor=C["green"]
        )
        self.btn_submit.label.set_fontsize(8.5)
        self.btn_submit.label.set_fontfamily("monospace")
        self.btn_submit.label.set_fontweight("bold")
        self.btn_submit.label.set_color(C["bg"])
        self.btn_submit.on_clicked(self._on_submit)

        self._switch_tab(0)

    def _switch_tab(self, tab_idx):
        """
        ---------------------------------------------------------------
        Function: _switch_tab
        Description:
            Shows only the content axes for the selected tab; hides
            all others. Updates tab button highlight colours.

        Parameters:
            tab_idx (int): Index of the tab to activate (0-5).
        ---------------------------------------------------------------
        """
        self.current_tab = tab_idx
        for i, ax in enumerate(self.content_axes):
            ax.set_visible(i == tab_idx)

        tab_labels = ["Control Panel", "City Map", "ANN Module",
                      "CSP Signals", "Pipeline", "Response"]
        for i, ax_tab in enumerate(self.tab_axes):
            ax_tab.clear()
            is_active = (i == tab_idx)
            ax_tab.set_facecolor(C["accent"] if is_active else C["panel2"])
            ax_tab.axis("off")
            if is_active:
                indicator = Rectangle((0, 0), 1, 0.06, facecolor=C["accent"],
                                      edgecolor="none", alpha=1.0, zorder=5,
                                      transform=ax_tab.transAxes, clip_on=False)
                ax_tab.add_patch(indicator)
            ax_tab.text(0.5, 0.5, tab_labels[i],
                        ha="center", va="center", fontsize=fs(7.5),
                        fontweight="bold",
                        color=C["bg"] if is_active else C["subtext"],
                        fontfamily="monospace",
                        transform=ax_tab.transAxes)

        self.fig.canvas.draw_idle()

    # ── Event handlers ──────────────────────────────────────────

    def _on_click(self, event):
        """
        ---------------------------------------------------------------
        Function: _on_click
        Description:
            Handles mouse click events on the figure. Detects clicks
            on tab buttons and switches to the selected tab.

        Parameters:
            event : matplotlib MouseEvent
        ---------------------------------------------------------------
        """
        for i, ax_tab in enumerate(self.tab_axes):
            if event.inaxes == ax_tab:
                self._switch_tab(i)
                return

    def _on_submit(self, event):
        """
        ---------------------------------------------------------------
        Function: _on_submit
        Description:
            Callback for the Submit button. Builds a request dict from
            current selections, runs the full AI pipeline, stores
            results, redraws all panels, and switches to the Response
            tab automatically.

        Parameters:
            event : matplotlib button click event (unused but required)
        ---------------------------------------------------------------
        """
        raw_data = {
            "request_id"       : str(uuid.uuid4())[:8].upper(),
            "request_category" : self.sel_category,
            "vehicle_type"     : self.sel_vehicle,
            "current_location" : self.sel_source,
            "destination"      : self.sel_dest,
            "incident_severity": self.sel_severity,
            "time_sensitivity" : self.sel_time,
            "traffic_density"  : self.sel_density,
            "priority_claim"   : self.sel_priority,
            "control_zone"     : self.sel_zone,
            "description_note" : ""
        }

        try:
            processed = self.preprocessor.process(raw_data)
            if not processed["valid"]:
                self._show_error(f"Validation Error:\n{processed['errors']}")
                return

            pipeline = self.router.route(processed)

            ann_r    = None
            logic_r  = None
            csp_r    = None
            search_r = None

            if "ANN"      in pipeline:
                ann_r    = self.ann.predict(processed)
            if "Logic_KB" in pipeline:
                logic_r  = self.logic_kb.validate(processed, ann_r)
            if "CSP"      in pipeline:
                csp_r    = self.csp.allocate(processed, logic_r)
            if "Search"   in pipeline:
                search_r = self.search.find_route(processed, csp_r)

            self.last_results = {
                "request"      : processed,
                "pipeline"     : pipeline,
                "ann_result"   : ann_r,
                "logic_result" : logic_r,
                "csp_result"   : csp_r,
                "search_result": search_r
            }
            self.last_response = self.response_layer.generate(self.last_results)

            self._draw_all()
            self._switch_tab(5)

        except Exception as exc:
            self._show_error(f"Pipeline error:\n{exc}")

    def _show_error(self, msg):
        """
        ---------------------------------------------------------------
        Function: _show_error
        Description:
            Displays an error message on the response panel.

        Parameters:
            msg (str): Error message text to display.
        ---------------------------------------------------------------
        """
        self.ax_response.clear()
        self.ax_response.set_facecolor(C["bg"])
        self.ax_response.axis("off")
        self.ax_response.text(
            0.5, 0.5, msg, ha="center", va="center",
            fontsize=fs(9), color=C["red"], fontfamily="monospace",
            transform=self.ax_response.transAxes,
            bbox=dict(facecolor=C["panel2"], edgecolor=C["red"],
                      boxstyle="round,pad=0.6", linewidth=1.5)
        )
        self._switch_tab(5)

    # ── Draw all panels ─────────────────────────────────────────

    def _draw_all(self):
        """
        ---------------------------------------------------------------
        Function: _draw_all
        Description:
            Redraws every panel (control panel, city map, ANN, CSP,
            pipeline, response) using the latest stored results.
            Called after each successful submission and at startup.
        ---------------------------------------------------------------
        """
        self._draw_control_panel()
        self._draw_map()
        self._draw_ann()
        self._draw_csp()
        self._draw_pipeline()
        self._draw_response()

    def _draw_control_panel(self):
        """
        ---------------------------------------------------------------
        Function: _draw_control_panel
        Description:
            Renders the Control Panel tab — a visual form showing all
            current request selections as labelled value badges, with
            instructions on how to change them (click to cycle).
        ---------------------------------------------------------------
        """
        ax = self.ax_control
        ax.clear()
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Subtle grid background
        for gy in np.arange(0, 10.5, 0.9):
            ax.axhline(gy, color="#1a2636", linewidth=0.6, alpha=0.5, zorder=0)

        

        fields = [
            ("Category",         self.sel_category,      CATEGORIES,       "sel_category"),
            ("Vehicle Type",     self.sel_vehicle,        VEHICLE_TYPES,    "sel_vehicle"),
            ("Source Location",  self.sel_source,         VALID_LOCATIONS,  "sel_source"),
            ("Destination",      self.sel_dest,           VALID_LOCATIONS,  "sel_dest"),
            ("Severity",         self.sel_severity,       SEVERITIES,       "sel_severity"),
            ("Traffic Density",  self.sel_density,        DENSITIES,        "sel_density"),
            ("Control Zone",     self.sel_zone,           VALID_ZONES,      "sel_zone"),
            ("Time Sensitive",   str(self.sel_time),      ["True", "False"], "sel_time"),
            ("Priority Claim",   str(self.sel_priority),  ["True", "False"], "sel_priority"),
        ]

        self._field_rows = fields

        row_h   = 0.80
        start_y = 8.7
        col1_x, col2_x, col3_x = 0.5, 4.5, 8.0

        # Column headers
        for hx, ht in [(col1_x, "FIELD"), (col2_x, "CURRENT VALUE"), (col3_x, "OPTIONS")]:
            ax.text(hx, start_y + 0.22, ht, fontsize=fs(7.5), color=C["accent2"],
                    fontfamily="monospace", fontweight="bold", zorder=2)

        # Separator line
        ax.plot([0.3, 13.7], [start_y + 0.08, start_y + 0.08],
                color=C["border"], linewidth=0.8, alpha=0.7, zorder=2)

        self._field_y_coords = []

        # Severity → colour mapping for badge accents
        sev_col = {"Critical": C["red"], "High": C["yellow"],
                   "Medium": C["accent"], "Low": C["green"],
                   "True": C["green"], "False": C["subtext"]}

        for idx, (fname, fval, fopts, fattr) in enumerate(fields):
            y = start_y - (idx + 1) * row_h

            row_bg = FancyBboxPatch(
                (0.22, y - 0.12), 13.56, row_h - 0.06,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor=C["panel2"] if idx % 2 == 0 else C["panel"],
                edgecolor=C["border2"], linewidth=0.5, alpha=0.85, zorder=1
            )
            ax.add_patch(row_bg)
            self._field_y_coords.append(
                (y - 0.12, y - 0.12 + row_h - 0.06, fattr, fopts, fval)
            )

            # Row index indicator on the left
            ax.text(0.35, y + 0.22, f"{idx + 1}", fontsize=fs(6.5),
                    color=C["subtext"], fontfamily="monospace",
                    ha="center", zorder=2)

            ax.text(col1_x, y + 0.22, fname, fontsize=fs(8), color=C["text"],
                    fontfamily="monospace", zorder=2)

            # Value badge — FIXED width so long values never bleed into OPTIONS
            val_accent   = sev_col.get(fval, C["accent"])
            badge_w      = 3.0          # fixed; wide enough for longest value
            fval_display = fval if len(fval) <= 26 else fval[:24] + "…"
            val_box = FancyBboxPatch(
                (col2_x - 0.12, y + 0.04), badge_w, 0.40,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=C["bg"], edgecolor=val_accent,
                linewidth=1.2, zorder=2
            )
            ax.add_patch(val_box)
            ax.text(col2_x + badge_w / 2 - 0.12, y + 0.26, fval_display,
                    fontsize=fs(7.5), color=val_accent, fontweight="bold",
                    fontfamily="monospace", zorder=3,
                    ha="center", va="center")

            # Options preview — stack vertically in two lines to avoid overflow
            opts_start_x = col2_x - 0.12 + badge_w + 0.45
            avail_w      = 13.5 - opts_start_x   # usable x-space remaining

            def short(name, maxlen=14):
                """Truncate long option names for display."""
                return name if len(name) <= maxlen else name[:maxlen - 1] + "…"

            # Split options across two lines
            half      = max(1, len(fopts) // 2 + len(fopts) % 2)
            line1_opts = fopts[:half]
            line2_opts = fopts[half:]

            line1 = "  |  ".join(f"[{i+1}] {short(o)}" for i, o in enumerate(line1_opts))
            line2 = "  |  ".join(
                f"[{i+1+half}] {short(o)}" for i, o in enumerate(line2_opts)
            )
            if len(fopts) > 10:
                line2 += "  …"

            ax.text(opts_start_x, y + 0.38, line1, fontsize=fs(5.6),
                    color=C["subtext"], fontfamily="monospace", zorder=2,
                    va="center")
            if line2:
                ax.text(opts_start_x, y + 0.12, line2, fontsize=fs(5.6),
                        color=C["dim"], fontfamily="monospace", zorder=2,
                        va="center")

        # Footer hint
        hint_box = FancyBboxPatch((1.5, 0.10), 11.0, 0.48,
                                  boxstyle="round,pad=0.05,rounding_size=0.12",
                                  facecolor=C["panel2"], edgecolor=C["border"],
                                  linewidth=0.8, zorder=2)
        ax.add_patch(hint_box)
        ax.text(7, 0.36,
                "Click on a row to cycle to the next option   |   "
                "Then press  ▶ SUBMIT REQUEST  to run the pipeline",
                ha="center", fontsize=fs(7.5), color=C["subtext"],
                fontfamily="monospace", zorder=3)

        if not hasattr(self, "_cp_click_connected"):
            self.fig.canvas.mpl_connect("button_press_event", self._on_control_click)
            self._cp_click_connected = True

    def _on_control_click(self, event):
        """
        ---------------------------------------------------------------
        Function: _on_control_click
        Description:
            Handles mouse clicks on Control Panel field rows. Cycles
            the clicked field to its next option in the list.

        Parameters:
            event : matplotlib MouseEvent
        ---------------------------------------------------------------
        """
        if event.inaxes != self.ax_control:
            return
        if not hasattr(self, "_field_y_coords"):
            return

        attr_map = {
            "sel_category": CATEGORIES,
            "sel_vehicle" : VEHICLE_TYPES,
            "sel_source"  : VALID_LOCATIONS,
            "sel_dest"    : VALID_LOCATIONS,
            "sel_severity": SEVERITIES,
            "sel_density" : DENSITIES,
            "sel_zone"    : VALID_ZONES,
            "sel_time"    : [True, False],
            "sel_priority": [True, False],
        }

        y_click = event.ydata
        if y_click is None:
            return

        for (y_bot, y_top, fattr, fopts, fval) in self._field_y_coords:
            if y_bot <= y_click <= y_top:
                opts    = attr_map.get(fattr, fopts)
                cur_val = getattr(self, fattr)
                try:
                    cur_idx = opts.index(cur_val)
                except ValueError:
                    try:
                        cur_idx = [str(o) for o in opts].index(str(cur_val))
                    except ValueError:
                        cur_idx = 0
                next_val = opts[(cur_idx + 1) % len(opts)]
                setattr(self, fattr, next_val)
                self._draw_control_panel()
                self.fig.canvas.draw_idle()
                break

    def _draw_map(self):
        """
        ---------------------------------------------------------------
        Function: _draw_map
        Description:
            Clears and redraws the City Map panel with the latest
            route highlighted (if any).
        ---------------------------------------------------------------
        """
        ax = self.ax_map
        ax.clear()
        route_path   = None
        is_emergency = False
        if self.last_results:
            sr = self.last_results.get("search_result")
            if sr and sr.get("found"):
                route_path = sr["path"]
            is_emergency = self.last_results["request"].get("is_emergency_vehicle", False)
        draw_city_map(ax, route_path, is_emergency)

    def _draw_ann(self):
        """
        ---------------------------------------------------------------
        Function: _draw_ann
        Description:
            Clears and redraws the ANN Module panel with the latest
            priority prediction results (if any).
        ---------------------------------------------------------------
        """
        ax = self.ax_ann
        ax.clear()
        ann_r = self.last_results.get("ann_result") if self.last_results else None
        draw_ann_diagram(ax, ann_r)

    def _draw_csp(self):
        """
        ---------------------------------------------------------------
        Function: _draw_csp
        Description:
            Clears and redraws the CSP Signals panel with the latest
            signal allocation (if any).
        ---------------------------------------------------------------
        """
        ax = self.ax_csp
        ax.clear()
        csp_r = self.last_results.get("csp_result") if self.last_results else None
        draw_csp_grid(ax, csp_r)

    def _draw_pipeline(self):
        """
        ---------------------------------------------------------------
        Function: _draw_pipeline
        Description:
            Clears and redraws the Pipeline Flow panel showing which
            modules ran for the latest request.
        ---------------------------------------------------------------
        """
        ax = self.ax_pipeline
        ax.clear()
        if self.last_results:
            draw_pipeline(ax, self.last_results["pipeline"], self.last_results)
        else:
            draw_pipeline(ax, None, None)

    def _draw_response(self):
        """
        ---------------------------------------------------------------
        Function: _draw_response
        Description:
            Clears and redraws the Final Response panel with the
            latest aggregated system response.
        ---------------------------------------------------------------
        """
        ax = self.ax_response
        ax.clear()
        draw_response_panel(ax, self.last_response)




def main():
    """
    ---------------------------------------------------------------
    Function: main
    Description:
        Entry point for the GUI application. Creates the SmartCityGUI
        instance which initialises matplotlib and opens the window.
    ---------------------------------------------------------------
    """
    print("=" * 60)
    print("  Smart City Traffic & Emergency Response AI System")
    print("  Graphical Interface — powered by matplotlib")
    print("=" * 60)
    print("  Instructions:")
    print("  1. Use the CONTROL PANEL tab to set request fields")
    print("     (click on any row to cycle through options)")
    print("  2. Press ▶ SUBMIT REQUEST")
    print("  3. Explore results across all tabs:")
    print("     City Map | ANN Module | CSP Signals | Pipeline | Response")
    print("=" * 60)
    SmartCityGUI()


if __name__ == "__main__":
    main()
