# -*- coding: utf-8 -*-
"""
Interactive 3D Visualization for MNIST using Dash
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.manifold import TSNE
from io import BytesIO
from PIL import Image
import base64
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, no_update

# Import local modules
try:
    from dataloader import MNISTLoader
    from submission import PCA, GMM
    from autoencoder import AE
    from util import ae_encode, ensure_dir
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataloader import MNISTLoader
    from submission import PCA, GMM
    from autoencoder import AE
    from util import ae_encode, ensure_dir

# Global variables to store data
embeddings = {}
images = None
true_labels = None
cluster_labels = None
app = dash.Dash(__name__)

def numpy_to_b64(img_array):
    """Convert numpy image (28, 28) to base64 string"""
    # Normalize to 0-255 if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
        
    im = Image.fromarray(img_array)
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

def load_data(args):
    global images, true_labels, cluster_labels, embeddings
    
    # 1. Load Data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    dataset = loader.load()
    trainset = dataset["train"]
    
    raw_images = np.stack(trainset["image2D"])
    # Handle (N, 1, 28, 28) or (N, 28, 28)
    if raw_images.ndim == 4 and raw_images.shape[1] == 1:
        raw_images = raw_images.squeeze(1)
        
    raw_data_flat = np.stack(trainset["image1D"])
    raw_true_labels = trainset["label"]
    
    # 2. Sampling
    n_total = len(raw_images)
    if args.sample_size > 0 and n_total > args.sample_size:
        print(f"Sampling {args.sample_size} random points from {n_total}...")
        indices = np.random.choice(n_total, args.sample_size, replace=False)
        images = raw_images[indices]
        data_flat = raw_data_flat[indices]
        true_labels = raw_true_labels[indices]
    else:
        images = raw_images
        data_flat = raw_data_flat
        true_labels = raw_true_labels

    # 3. Compute Embeddings
    # PCA (3D)
    print("Computing PCA (3D)...")
    pca_3d = PCA_sklearn(n_components=3)
    embeddings['PCA'] = pca_3d.fit_transform(data_flat)
    
    # t-SNE (3D)
    print("Computing t-SNE (3D)...")
    tsne_3d = TSNE(n_components=3, init='pca', learning_rate='auto', random_state=42)
    embeddings['t-SNE'] = tsne_3d.fit_transform(data_flat)
    
    # AE (2D -> 3D)
    print("Computing Autoencoder (2D)...")
    try:
        ae = AE.from_pretrained("H2O123h2o/mnist-autoencoder")
        # ae_encode expects (N, 28, 28) or (N, 1, 28, 28)? 
        # util.ae_encode usually handles it.
        images_tensor = images[:, np.newaxis, :, :] if images.ndim == 3 else images
        ae_2d = ae_encode(ae, images_tensor) # Returns (N, 2)
        # Pad with zeros for 3D
        embeddings['Autoencoder'] = np.column_stack([ae_2d, np.zeros(len(ae_2d))])
    except Exception as e:
        print(f"Warning: Could not run Autoencoder: {e}")
        embeddings['Autoencoder'] = np.zeros((len(images), 3))

    # 4. Load Cluster Labels (if results_path provided)
    if args.results_path:
        print(f"Loading results from {args.results_path}...")
        try:
            config_path = Path(args.results_path) / "config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                dr_method = cfg.get('dr_method', 'pca')
                
                # Load GMM
                gmm_path = Path(args.results_path) / "gmm"
                if gmm_path.exists():
                    gmm = GMM.from_pretrained(str(gmm_path))
                    
                    if dr_method == 'pca':
                        print("Loading saved PCA model for clustering prediction...")
                        pca_path = Path(args.results_path) / "pca.npz"
                        if pca_path.exists():
                            pca_model = PCA.from_pretrained(str(pca_path))
                            # Transform the CURRENT sampled data
                            data_reduced = pca_model.transform(data_flat)
                            cluster_labels = gmm.predict(data_reduced)
                        else:
                            print("PCA model not found.")
                    else:
                        print("Using AE for clustering prediction...")
                        # We already computed ae_2d
                        cluster_labels = gmm.predict(ae_2d)
                else:
                    print("GMM model not found.")
            else:
                print("Warning: config.yaml not found. Cannot load clustering results.")
        except Exception as e:
            print(f"Error loading clustering results: {e}")

# App Layout
app.layout = html.Div([
    html.H1("MNIST 3D Interactive Visualization", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Dimensionality Reduction Method:"),
            dcc.Dropdown(
                id='method-dropdown',
                options=[
                    {'label': 'PCA', 'value': 'PCA'},
                    {'label': 't-SNE', 'value': 't-SNE'},
                    {'label': 'Autoencoder', 'value': 'Autoencoder'}
                ],
                value='PCA',
                clearable=False
            ),
            html.Br(),
            html.Label("Color By:"),
            dcc.RadioItems(
                id='color-radio',
                options=[
                    {'label': 'True Label', 'value': 'true'},
                    {'label': 'Cluster Label', 'value': 'cluster'}
                ],
                value='true'
            )
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            dcc.Graph(id='3d-scatter-plot', style={'height': '80vh'})
        ], style={'width': '55%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Data Point Details"),
            html.Div(id='hover-image-container', style={'textAlign': 'center'}),
            html.Div(id='hover-info', style={'marginTop': '20px'})
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f9f9f9'})
    ], style={'display': 'flex'})
])

@app.callback(
    Output('3d-scatter-plot', 'figure'),
    [Input('method-dropdown', 'value'),
     Input('color-radio', 'value')]
)
def update_graph(method, color_mode):
    data_3d = embeddings[method]
    
    if color_mode == 'cluster' and cluster_labels is not None:
        labels = cluster_labels
        title_color = "Cluster Label"
        colorscale = 'Viridis'
    else:
        labels = true_labels
        title_color = "True Label"
        colorscale = 'Jet'
        
    # Create customdata with index to retrieve image later
    # customdata: [index, true_label, cluster_label]
    c_labels = cluster_labels if cluster_labels is not None else np.full(len(labels), -1)
    customdata = np.column_stack([
        np.arange(len(labels)),
        true_labels,
        c_labels
    ])
        
    fig = go.Figure(data=[go.Scatter3d(
        x=data_3d[:, 0],
        y=data_3d[:, 1],
        z=data_3d[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=labels,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(title=title_color)
        ),
        customdata=customdata,
        hovertemplate="<b>Index:</b> %{customdata[0]}<br>" +
                      "<b>True Label:</b> %{customdata[1]}<br>" +
                      "<b>Cluster:</b> %{customdata[2]}<extra></extra>"
    )])
    
    fig.update_layout(
        title=f"Method: {method} | Color: {title_color}",
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

@app.callback(
    [Output('hover-image-container', 'children'),
     Output('hover-info', 'children')],
    [Input('3d-scatter-plot', 'hoverData')]
)
def display_hover_data(hoverData):
    if hoverData is None:
        return "Hover over a point to see details", ""
    
    # Get index from pointNumber (more reliable than customdata in callback)
    point_data = hoverData['points'][0]
    idx = point_data['pointNumber']
    
    # Retrieve labels from global data
    true_lbl = true_labels[idx]
    cluster_lbl = cluster_labels[idx] if cluster_labels is not None else -1
    
    # Get image
    img_b64 = numpy_to_b64(images[idx])
    
    img_elem = html.Img(src=img_b64, style={'width': '150px', 'height': '150px', 'border': '2px solid #333'})
    
    info_elem = html.Div([
        html.P(f"Index: {idx}"),
        html.P(f"True Label: {true_lbl}", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        html.P(f"Cluster Label: {cluster_lbl}" if cluster_lbl != -1 else "Cluster Label: N/A")
    ])
    
    return img_elem, info_elem

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Visualization of MNIST (Dash)")
    parser.add_argument("--results_path", type=str, default=None, help="Path to results directory (optional)")
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of samples to visualize")
    parser.add_argument("--plot_mode", type=str, default='comparison', choices=['clustering', 'comparison'], help="Visualization mode (ignored in Dash app, all modes available)")
    parser.add_argument("--port", type=int, default=8050, help="Port to run Dash app on")
    args = parser.parse_args()
    
    load_data(args)
    
    print(f"Starting Dash app on http://127.0.0.1:{args.port}/")
    app.run(debug=True, port=args.port, use_reloader=False)

if __name__ == "__main__":
    main()

