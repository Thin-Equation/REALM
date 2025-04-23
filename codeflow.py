#!/usr/bin/env python3
# codeflow.py - Visualization of REALM training process and codebase structure

import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_code_structure_graph():
    """Create a visualization of the REALM codebase structure using NetworkX and Matplotlib."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for different components with improved spacing
    components = {
        # Main module
        'Main': {'pos': (5, 12), 'color': '#ADD8E6', 'label': 'main.py\nEntrypoint'},
        
        # Core components - increased vertical spacing
        'Models': {'pos': (2, 9), 'color': '#90EE90', 'label': 'Models Module'},
        'RLHF': {'pos': (5, 9), 'color': '#90EE90', 'label': 'RLHF Module'},
        'Training': {'pos': (8, 9), 'color': '#90EE90', 'label': 'Training Module'},
        'Data': {'pos': (2, 6), 'color': '#90EE90', 'label': 'Data Processing'},
        'Inference': {'pos': (8, 6), 'color': '#90EE90', 'label': 'Inference Module'},
        
        # Models subcomponents - wider horizontal spacing
        'LinearRM': {'pos': (1, 3), 'color': '#FFB6C1', 'label': 'Linear Reward Model'},
        'NIMRM': {'pos': (3.5, 3), 'color': '#FFB6C1', 'label': 'NIM Reward Model'},
        
        # RLHF subcomponents - wider horizontal spacing
        'PPO': {'pos': (4, 3), 'color': '#FFFF99', 'label': 'PPO Trainer'},
        'DPO': {'pos': (7, 3), 'color': '#FFFF99', 'label': 'DPO Trainer'}
    }
    
    # Add nodes with positions
    for node, attrs in components.items():
        G.add_node(node, pos=attrs['pos'], color=attrs['color'], label=attrs['label'])
    
    # Add edges
    edges = [
        # Connections from main
        ('Main', 'Models'),
        ('Main', 'RLHF'),
        ('Main', 'Training'),
        ('Main', 'Data'),
        ('Main', 'Inference'),
        
        # Connections to model components
        ('Models', 'LinearRM'),
        ('Models', 'NIMRM'),
        
        # Connections to RLHF components
        ('RLHF', 'PPO'),
        ('RLHF', 'DPO'),
        
        # Integration connections
        ('LinearRM', 'PPO'),
        ('LinearRM', 'DPO'),
        ('NIMRM', 'LinearRM')
    ]
    
    G.add_edges_from(edges)
    
    # Create figure with larger size
    plt.figure(figsize=(14, 12))
    
    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes with different colors and larger size
    for node, attrs in components.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=attrs['color'], 
                              node_size=3500, alpha=0.9, node_shape='s', edgecolors='black')
    
    # Draw edges with curved paths to avoid overlap
    curved_edges = {
        ('Main', 'Models'): 0.2,
        ('Main', 'RLHF'): 0,
        ('Main', 'Training'): -0.2,
        ('Main', 'Data'): 0.3,
        ('Main', 'Inference'): -0.3,
        ('Models', 'LinearRM'): 0.1,
        ('Models', 'NIMRM'): -0.1,
        ('RLHF', 'PPO'): 0.1,
        ('RLHF', 'DPO'): -0.1,
        ('LinearRM', 'PPO'): 0.2,
        ('LinearRM', 'DPO'): 0.3,
        ('NIMRM', 'LinearRM'): 0.2
    }
    
    for edge in edges:
        rad = curved_edges.get(edge, 0)
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=1.5, arrowsize=20, 
                              edge_color='gray', connectionstyle=f'arc3,rad={rad}')
    
    # Draw labels with improved font sizes
    labels = {node: attrs['label'] for node, attrs in components.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold', font_family='sans-serif')
    
    # Draw edge labels with better positioning
    edge_labels = {
        ('LinearRM', 'PPO'): 'feeds rewards',
        ('LinearRM', 'DPO'): 'ranks preferences',
        ('NIMRM', 'LinearRM'): 'provides base rewards'
    }
    
    # Draw edge labels at the midpoint of edges with background for readability
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_size=10, 
        font_color='darkblue', 
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3),
        label_pos=0.5  # Position labels at midpoint of edges
    )
    
    plt.title('REALM Codebase Structure', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout(pad=1.5)
    
    # Save the figure with higher DPI
    plt.savefig('visualizations/realm_codebase_structure.png', dpi=400, bbox_inches='tight')
    return plt


def create_training_flow_diagram():
    """Create a visualization of the REALM training flow process using matplotlib."""
    # Create figure with larger size for better spacing
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define the three stages and their components with improved spacing
    stages = {
        "Stage 1: Linear NN Training": {
            "color": "#EEEEEE",
            "y": 8,  # More vertical spacing between stages
            "nodes": [
                {"name": "data_prep1", "label": "Prepare SHP Dataset\nwith Preference Pairs", "x": 1, "color": "#ADD8E6"},
                {"name": "nim_rewards", "label": "Extract NIM\nReward Scores", "x": 3, "color": "#ADD8E6"},
                {"name": "embed_sim", "label": "Calculate Embedding\nSimilarities", "x": 3, "y_offset": -1.5, "color": "#ADD8E6"},  # More vertical offset
                {"name": "linear_train", "label": "Train Linear NN\nto Combine Signals", "x": 5.5, "color": "#90EE90"},  # More horizontal spacing
                {"name": "linear_eval", "label": "Evaluate Linear NN\non Validation Set", "x": 8, "color": "#90EE90"},
                {"name": "linear_ckpt", "label": "Save Best\nModel Weights", "x": 10.5, "color": "#FFFF99"}  # More horizontal spacing
            ],
            "edges": [
                ("data_prep1", "nim_rewards"),
                ("data_prep1", "embed_sim"),
                ("nim_rewards", "linear_train"),
                ("embed_sim", "linear_train"),
                ("linear_train", "linear_eval"),
                ("linear_eval", "linear_ckpt")
            ]
        },
        "Stage 2: RLHF Training on together.ai": {
            "color": "#E6F3FF",
            "y": 4.5,  # More vertical spacing
            "nodes": [
                {"name": "data_prep2", "label": "Prepare Stanford\nDataset", "x": 1, "color": "#B0E0E6"},
                {"name": "reward_setup", "label": "Configure Linear NN\nas Reward Model", "x": 3, "color": "#B0E0E6"},
                {"name": "ppo_train", "label": "Train with PPO", "x": 5.5, "y_offset": 0.7, "color": "#6495ED"},  # Vertical offset to separate
                {"name": "dpo_train", "label": "Train with DPO", "x": 5.5, "y_offset": -0.7, "color": "#6495ED"},  # Vertical offset to separate
                {"name": "rlhf_ckpt", "label": "Save RLHF\nModel Checkpoints", "x": 8, "color": "#FFA500"}
            ],
            "edges": [
                ("data_prep2", "ppo_train"),
                ("data_prep2", "dpo_train"),
                ("reward_setup", "ppo_train"),
                ("reward_setup", "dpo_train"),
                ("ppo_train", "rlhf_ckpt"),
                ("dpo_train", "rlhf_ckpt")
            ]
        },
        "Stage 3: Final Evaluation": {
            "color": "#FFEFF0",
            "y": 1,
            "nodes": [
                {"name": "qa_prep", "label": "Prepare Truthfulness\nQA Dataset", "x": 1, "color": "#F08080"},
                {"name": "model_eval", "label": "Evaluate Tuned Models\non QA Dataset", "x": 3, "color": "#F08080"},
                {"name": "metrics", "label": "Calculate Metrics\n(Accuracy, F1)", "x": 5.5, "color": "#CD5C5C"},
                {"name": "compare", "label": "Compare Base vs.\nTuned Models", "x": 8, "color": "#CD5C5C"},
                {"name": "final_report", "label": "Generate Final\nPerformance Report", "x": 10.5, "color": "#B22222"}
            ],
            "edges": [
                ("qa_prep", "model_eval"),
                ("model_eval", "metrics"),
                ("metrics", "compare"),
                ("compare", "final_report")
            ]
        }
    }
    
    # Prepare node positions and names for drawing
    node_positions = {}
    node_colors = {}
    node_labels = {}
    
    # Process each stage and its nodes
    for stage_name, stage in stages.items():
        # Draw stage background rectangle with rounded corners
        stage_y = stage["y"]
        stage_height = 2.4  # Taller rectangles for more space
        stage_width = 11.5  # Wider rectangles for more space
        
        # Create a rectangle with shadow effect
        rect = plt.Rectangle((0.3, stage_y - stage_height/2), stage_width, stage_height, 
                           color=stage["color"], alpha=0.7, ec="gray", lw=1.5,
                           zorder=1)
        
        # Add shadow effect (slight offset black rectangle)
        shadow = plt.Rectangle((0.4, stage_y - stage_height/2 - 0.1), stage_width, stage_height, 
                             color="gray", alpha=0.2, zorder=0)
        ax.add_patch(shadow)
        ax.add_patch(rect)
        
        # Add stage label with better styling
        ax.text(0.7, stage_y + 0.9, stage_name, fontsize=16, weight='bold', ha='left', va='center',
               bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Process nodes in this stage
        for node in stage["nodes"]:
            x = node["x"]
            y = stage_y + node.get("y_offset", 0)
            node_name = node["name"]
            node_positions[node_name] = (x, y)
            node_colors[node_name] = node["color"]
            node_labels[node_name] = node["label"]
    
    # Add cross-stage connections
    cross_stage_edges = [
        ("linear_ckpt", "reward_setup"),
        ("rlhf_ckpt", "model_eval")
    ]
    
    # Draw all nodes - bigger size and with shadow effect
    for node_name, pos in node_positions.items():
        # Add shadow
        shadow = plt.Circle((pos[0]+0.05, pos[1]-0.05), 0.5, color="gray", alpha=0.3, zorder=9)
        ax.add_patch(shadow)
        
        # Add node with larger radius (0.5 instead of 0.4)
        circle = plt.Circle(pos, 0.5, color=node_colors[node_name], ec="dimgray", 
                           lw=1.5, zorder=10, alpha=0.9)
        ax.add_patch(circle)
        
        # Add text with background for better readability
        text_obj = ax.text(pos[0], pos[1], node_labels[node_name], ha='center', va='center', 
                          fontsize=10, weight='bold', zorder=11,
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', 
                                     edgecolor='none'))
    
    # Draw all edges within stages with controlled curvature
    edge_curves = {
        # Linear Training
        ("data_prep1", "nim_rewards"): 0.1,
        ("data_prep1", "embed_sim"): -0.1,
        ("nim_rewards", "linear_train"): 0.1,
        ("embed_sim", "linear_train"): -0.1,
        ("linear_train", "linear_eval"): 0,
        ("linear_eval", "linear_ckpt"): 0,
        
        # RLHF Training
        ("data_prep2", "ppo_train"): 0.2,
        ("data_prep2", "dpo_train"): -0.2,
        ("reward_setup", "ppo_train"): 0.2,
        ("reward_setup", "dpo_train"): -0.2,
        ("ppo_train", "rlhf_ckpt"): 0.1,
        ("dpo_train", "rlhf_ckpt"): -0.1,
        
        # Evaluation
        ("qa_prep", "model_eval"): 0,
        ("model_eval", "metrics"): 0,
        ("metrics", "compare"): 0,
        ("compare", "final_report"): 0
    }
    
    # Draw all stage internal edges
    for stage_name, stage in stages.items():
        for edge in stage["edges"]:
            source, target = edge
            start_pos = node_positions[source]
            end_pos = node_positions[target]
            
            # Calculate positions on the circle edges accounting for larger radius (0.5)
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist == 0:
                continue  # Skip self-loops
            nx = dx / dist
            ny = dy / dist
            start_x = start_pos[0] + 0.5 * nx
            start_y = start_pos[1] + 0.5 * ny
            end_x = end_pos[0] - 0.5 * nx
            end_y = end_pos[1] - 0.5 * ny
            
            # Get curvature from dictionary or use default
            rad = edge_curves.get(edge, 0)
            
            # Create arrow
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                                  arrowstyle='-|>', color='black', lw=2, 
                                  connectionstyle=f'arc3,rad={rad}', zorder=5)
            ax.add_patch(arrow)
    
    # Draw cross-stage edges with distinctive style
    for source, target in cross_stage_edges:
        start_pos = node_positions[source]
        end_pos = node_positions[target]
        
        # Create longer arrow for cross-stage connections with clear path
        arrow = FancyArrowPatch((start_pos[0], start_pos[1]-0.5), (end_pos[0], end_pos[1]+0.5), 
                             arrowstyle='-|>', color='red', lw=2.5, 
                             connectionstyle='arc3,rad=-0.4', zorder=5,
                             linestyle='dashed')
        ax.add_patch(arrow)
        
        # Add a label to the cross-stage connections
        mid_x = (start_pos[0] + end_pos[0]) / 2 - 0.5
        mid_y = (start_pos[1] + end_pos[1]) / 2
        if source == "linear_ckpt":
            label = "Trained Weights"
        else:
            label = "Model Checkpoint"
        
        # Add text with background
        ax.text(mid_x, mid_y, label, fontsize=10, ha='center', va='center', 
               color='darkred', weight='bold', rotation=-45,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Set limits and remove ticks
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title with better styling
    plt.suptitle('REALM Training Flow Diagram', fontsize=20, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.01, 'REALM: Reward Enhanced Alignment Learning Method', 
               ha='center', fontsize=14, fontstyle='italic')
    
    plt.tight_layout(pad=2.5)
    
    # Save the figure with higher DPI
    plt.savefig('visualizations/realm_training_flow.png', dpi=400, bbox_inches='tight')
    return plt


def create_detailed_architecture():
    """Create a visual representation of the REALM architecture with better spacing and clarity."""
    # Create a figure with a larger size for better spacing
    plt.figure(figsize=(16, 12))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for different components with improved spacing
    components = {
        # Data nodes with better vertical separation
        'SHP_Dataset': {'pos': (1, 9), 'color': '#ADD8E6', 'shape': 'o'},
        'Stanford_Dataset': {'pos': (1, 5.5), 'color': '#ADD8E6', 'shape': 'o'},
        'QA_Dataset': {'pos': (1, 2), 'color': '#ADD8E6', 'shape': 'o'},
        
        # Model nodes with better separation
        'NIM_Model': {'pos': (4, 10), 'color': '#90EE90', 'shape': 's'},
        'Embedding_Model': {'pos': (4, 8), 'color': '#90EE90', 'shape': 's'},
        
        # Linear NN at central position
        'Linear_NN': {'pos': (7, 9), 'color': '#FFFF99', 'shape': 'D'},
        
        # Training nodes with better separation
        'PPO_Training': {'pos': (10, 7), 'color': '#FFA500', 'shape': 's'},
        'DPO_Training': {'pos': (10, 4), 'color': '#FFA500', 'shape': 's'},
        
        # Results nodes
        'Tuned_Model': {'pos': (13, 5.5), 'color': '#FFB6C1', 'shape': 'D'},
        'Evaluation': {'pos': (16, 3), 'color': '#F08080', 'shape': 's'},
        'Final_Results': {'pos': (19, 3), 'color': '#B22222', 'shape': 'o'}
    }
    
    # Add nodes with positions, colors, and shapes
    for node, attrs in components.items():
        G.add_node(node, pos=attrs['pos'], color=attrs['color'], shape=attrs['shape'])
    
    # Add edges with specific connection styles to avoid overlap
    edge_styles = {
        ('SHP_Dataset', 'NIM_Model'): {'connectionstyle': 'arc3,rad=0.1'},
        ('SHP_Dataset', 'Embedding_Model'): {'connectionstyle': 'arc3,rad=-0.1'},
        ('NIM_Model', 'Linear_NN'): {'connectionstyle': 'arc3,rad=0.1'},
        ('Embedding_Model', 'Linear_NN'): {'connectionstyle': 'arc3,rad=-0.1'},
        ('Linear_NN', 'PPO_Training'): {'connectionstyle': 'arc3,rad=0.1'},
        ('Linear_NN', 'DPO_Training'): {'connectionstyle': 'arc3,rad=-0.2'},
        ('Stanford_Dataset', 'PPO_Training'): {'connectionstyle': 'arc3,rad=0.2'},
        ('Stanford_Dataset', 'DPO_Training'): {'connectionstyle': 'arc3,rad=-0.1'},
        ('PPO_Training', 'Tuned_Model'): {'connectionstyle': 'arc3,rad=0.1'},
        ('DPO_Training', 'Tuned_Model'): {'connectionstyle': 'arc3,rad=-0.1'},
        ('Tuned_Model', 'Evaluation'): {'connectionstyle': 'arc3,rad=0'},
        ('QA_Dataset', 'Evaluation'): {'connectionstyle': 'arc3,rad=0.3'},
        ('Evaluation', 'Final_Results'): {'connectionstyle': 'arc3,rad=0'}
    }
    
    # Add all edges
    edges = list(edge_styles.keys())
    G.add_edges_from(edges)
    
    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes with different shapes and sizes based on their role
    shapes = nx.get_node_attributes(G, 'shape')
    for shape_type in set(shapes.values()):
        # Filter nodes by shape
        nodelist = [node for node, shape in shapes.items() if shape == shape_type]
        
        # Get colors for these nodes
        node_colors = [components[node]['color'] for node in nodelist]
        
        # Set specific shape and size
        if shape_type == 'o':  # Circle for data
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors,
                                  node_size=4000, alpha=0.9, node_shape='o', edgecolors='black', linewidths=2)
        elif shape_type == 's':  # Square for processing
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors,
                                  node_size=4000, alpha=0.9, node_shape='s', edgecolors='black', linewidths=2)
        elif shape_type == 'D':  # Diamond for key models
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors,
                                  node_size=5000, alpha=0.9, node_shape='D', edgecolors='black', linewidths=2.5)
    
    # Draw edges with custom styles
    for edge, style in edge_styles.items():
        # Draw each edge with its specific style
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2.0, arrowsize=25, 
                              alpha=0.8, edge_color='dimgray', style='solid',
                              connectionstyle=style['connectionstyle'], min_source_margin=20, min_target_margin=20)
    
    # Better labels with backgrounds for readability
    labels = {
        'SHP_Dataset': 'Stanford Human\nPreferences Dataset',
        'Stanford_Dataset': 'Stanford Dataset\nfor RLHF',
        'QA_Dataset': 'Truthfulness\nQA Dataset',
        'NIM_Model': 'NIM Reward\nModel',
        'Embedding_Model': 'Embedding\nSimilarity',
        'Linear_NN': 'Linear Neural Network\nReward Combiner',
        'PPO_Training': 'PPO Training\non together.ai',
        'DPO_Training': 'DPO Training\non together.ai',
        'Tuned_Model': 'Fine-tuned\nLanguage Model',
        'Evaluation': 'Performance\nEvaluation',
        'Final_Results': 'Final Metrics\n& Analysis'
    }
    
    # Add text with white background for readability
    label_positions = {}
    # Apply slight offset to labels
    for node in G.nodes():
        x, y = pos[node]
        # Apply an offset to avoid overlap with node shapes
        label_positions[node] = (x, y)
    
    # Draw labels with background boxes
    for node, label in labels.items():
        plt.text(label_positions[node][0], label_positions[node][1], label,
               fontsize=11, fontweight='bold', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
               zorder=30)  # Higher zorder to ensure text is on top
    
    # Add title with better styling
    plt.suptitle('REALM: Reward Enhanced Alignment Learning Method', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add subtitles for the different sections
    plt.text(1, 11.5, "Data Sources", fontsize=16, fontweight='bold', ha='center',
            bbox=dict(facecolor='#E6F3FF', alpha=0.6, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    plt.text(5.5, 11.5, "Model Components", fontsize=16, fontweight='bold', ha='center',
            bbox=dict(facecolor='#E6FFE6', alpha=0.6, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    plt.text(10, 11.5, "Training Process", fontsize=16, fontweight='bold', ha='center',
            bbox=dict(facecolor='#FFF9E6', alpha=0.6, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    plt.text(16, 11.5, "Evaluation", fontsize=16, fontweight='bold', ha='center',
            bbox=dict(facecolor='#FFE6E6', alpha=0.6, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    # Add flow description at bottom
    plt.figtext(0.5, 0.01, 'Data → Models → Training → Evaluation Pipeline', 
              ha='center', fontsize=14, fontstyle='italic')
    
    # Draw stage separating lines (vertical lines to show stages)
    stage_xs = [3, 8.5, 14.5]
    for x in stage_xs:
        plt.axvline(x=x, ymin=0.05, ymax=0.92, color='gray', linestyle='--', alpha=0.5)
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    
    # Save the figure with higher quality
    plt.savefig('visualizations/realm_detailed_architecture.png', dpi=400, bbox_inches='tight')
    return plt


def generate_training_phase_diagram():
    """Generate a professional training phase diagram as a PNG image with improved layout."""
    # Create a white image with larger dimensions for better spacing
    width, height = 1600, 1000
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw subtle background gradient for visual interest
    for y in range(height):
        # Create subtle gradient from very light blue to white
        color_value = int(240 + (255 - 240) * (y / height))
        line_color = (color_value, color_value, 255)
        draw.line([(0, y), (width, y)], fill=line_color)
    
    try:
        # Try to load a font - fallback to default if not available
        font_title = ImageFont.truetype('Arial', 48)  # Larger title
        font_subtitle = ImageFont.truetype('Arial', 28)
        font_header = ImageFont.truetype('Arial', 28)  # Larger headers
        font_text = ImageFont.truetype('Arial', 18)  # Slightly larger text
        font_note = ImageFont.truetype('Arial', 20)  # Font for notes
    except IOError:
        # Fallback to default font
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_note = ImageFont.load_default()
    
    # Draw main title with shadow effect for depth
    title_text = "REALM Training Process Flow"
    # Shadow
    draw.text((width//2+3, 43), title_text, fill="#AAAAAA", font=font_title, anchor="mm")
    # Main text
    draw.text((width//2, 40), title_text, fill="#000066", font=font_title, anchor="mm")
    
    # Add subtitle
    draw.text((width//2, 90), "Reward Enhanced Alignment Learning Method", 
             fill="#444444", font=font_subtitle, anchor="mm")
    
    # Define phases with more vertical spacing between them
    phases = [
        {"name": "Stage 1: Linear Neural Network Training", "color": "#CCFFCC", "y": 200, "steps": [
            "Load SHP Dataset with paired responses",
            "Extract NIM reward scores",
            "Calculate embedding similarities",
            "Train linear NN to combine signals",
            "Save model weights"
        ]},
        {"name": "Stage 2: PPO Training on together.ai", "color": "#CCE5FF", "y": 420, "steps": [
            "Configure linear NN as reward model",
            "Prepare Stanford dataset",
            "Set PPO hyperparameters",
            "Run training on together.ai",
            "Monitor and save checkpoints"
        ]},
        {"name": "Stage 3: DPO Training on together.ai", "color": "#FFD6E0", "y": 640, "steps": [
            "Configure linear NN for preference ranking",
            "Format paired responses",
            "Set DPO hyperparameters",
            "Run training on together.ai",
            "Save best model checkpoint"
        ]},
        {"name": "Stage 4: Evaluation on Truthfulness QA", "color": "#FFFACD", "y": 860, "steps": [
            "Load tuned model weights",
            "Prepare truthfulness QA dataset",
            "Run inference and collect predictions",
            "Calculate accuracy and other metrics",
            "Compare against baseline model"
        ]}
    ]
    
    # Draw phases with improved aesthetics
    for phase in phases:
        # Draw phase box with more space
        box_left = 120
        box_right = width - 120
        box_top = phase["y"] - 70  # Taller boxes
        box_bottom = phase["y"] + 70
        
        # Draw box shadow (slight offset)
        shadow_offset = 8
        draw.rectangle([box_left+shadow_offset, box_top+shadow_offset, 
                        box_right+shadow_offset, box_bottom+shadow_offset], 
                       fill="#DDDDDD", outline=None)
        
        # Draw rounded rectangle with thicker outline
        draw.rounded_rectangle([box_left, box_top, box_right, box_bottom], 
                               radius=20, fill=phase["color"], 
                               outline="#666666", width=2)
        
        # Draw phase name with slight shadow for depth
        # Shadow
        draw.text((width//2+2, phase["y"]+2), phase["name"], 
                 fill="#666666", font=font_header, anchor="mm")
        # Text
        draw.text((width//2, phase["y"]), phase["name"], 
                 fill="#000066", font=font_header, anchor="mm")
        
        # Draw steps with more space between them
        num_steps = len(phase["steps"])
        usable_width = box_right - box_left - 100  # Leave margin on both sides
        step_width = usable_width / num_steps
        
        for i, step in enumerate(phase["steps"]):
            step_x = box_left + 50 + step_width/2 + i * step_width  # Start 50px from left edge
            step_y = phase["y"] + 110  # More distance from phase name
            
            # Draw step number in a circle
            circle_radius = 15
            draw.ellipse([step_x-circle_radius, step_y-circle_radius-35, 
                          step_x+circle_radius, step_y+circle_radius-35], 
                         fill="#FFFFFF", outline="#000000")
            draw.text((step_x, step_y-35), f"{i+1}", fill="#000000", font=font_text, anchor="mm")
            
            # Draw step text with background for readability
            text_width = draw.textlength(step, font=font_text)
            # Create text background
            draw.rounded_rectangle([step_x-text_width/2-10, step_y-15, 
                                    step_x+text_width/2+10, step_y+15], 
                                   radius=10, fill="#FFFFFF", outline="#AAAAAA")
            # Draw text
            draw.text((step_x, step_y), step, fill="#000000", font=font_text, anchor="mm")
    
    # Draw connecting arrows between phases with improved styling
    for i in range(len(phases)-1):
        arrow_start_y = phases[i]["y"] + 70  # Bottom of current phase box
        arrow_end_y = phases[i+1]["y"] - 70  # Top of next phase box
        
        # Arrow shaft with gradient effect (thicker at top, thinner at bottom)
        for offset in range(7):
            # Draw multiple lines with decreasing opacity
            opacity = 255 - (offset * 30)  # Decrease opacity for each offset
            if opacity < 0:
                opacity = 0
            arrow_color = (0, 0, 100, opacity)  # RGBA with varying alpha
            width_value = 7 - offset  # Decrease width for each offset
            if width_value > 0:
                draw.line([(width//2-offset, arrow_start_y), (width//2-offset, arrow_end_y)], 
                         fill=arrow_color, width=width_value)
        
        # Main arrow line
        draw.line([(width//2, arrow_start_y), (width//2, arrow_end_y)], 
                 fill="#000066", width=3)
        
        # Draw larger, more visible arrowhead
        arrowhead_size = 15
        draw.polygon([(width//2, arrow_end_y), 
                      (width//2-arrowhead_size, arrow_end_y-arrowhead_size*1.5), 
                      (width//2+arrowhead_size, arrow_end_y-arrowhead_size*1.5)], 
                     fill="#000066")
        
        # Add transition label
        mid_y = (arrow_start_y + arrow_end_y) // 2
        if i == 0:
            label = "Linear NN Weights"
        elif i == 1:
            label = "Training Configuration"
        else:
            label = "Model Checkpoint"
        
        # White background for label with border
        text_width = draw.textlength(label, font=font_note)
        draw.rounded_rectangle([width//2-text_width/2-10, mid_y-15, 
                                width//2+text_width/2+10, mid_y+15], 
                               radius=10, fill="#FFFFFF", outline="#666666")
        
        # Draw label
        draw.text((width//2, mid_y), label, fill="#000066", font=font_note, anchor="mm")
    
    # Add footer note
    footer_text = "The complete REALM training pipeline represents a sequential flow from initial model training to final evaluation."
    draw.text((width//2, height-50), footer_text, fill="#444444", font=font_note, anchor="mm")
    
    # Save the image with higher quality
    image.save('realm_training_phases.png', quality=95, dpi=(300, 300))
    return 'realm_training_phases.png'


def main():
    """Generate all visualizations."""
    # Create directory for saving visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate the code structure graph
    create_code_structure_graph()
    print("Code structure visualization saved as 'visualizations/realm_codebase_structure.png'")
    plt.close()
    
    # Generate the training flow diagram
    create_training_flow_diagram()
    print("Training flow visualization saved as 'visualizations/realm_training_flow.png'")
    plt.close()
    
    # Generate detailed architecture diagram
    arch_plt = create_detailed_architecture()
    plt.close()
    print("Detailed architecture saved as 'visualizations/realm_detailed_architecture.png'")
    
    # Generate training phase diagram
    phase_diagram = generate_training_phase_diagram()
    os.rename(phase_diagram, 'visualizations/realm_training_phases.png')
    print("Training phases diagram saved as 'visualizations/realm_training_phases.png'")
    
    print("\nAll visualizations have been generated in the 'visualizations' directory.")
    print("You'll need to install matplotlib, networkx, and PIL if you haven't already:")
    print("pip install matplotlib networkx pillow")
    

if __name__ == "__main__":
    main()
