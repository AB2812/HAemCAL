import os
from threading import Thread
import tkinter as tk
from tkinter import  ttk, filedialog, messagebox
import webbrowser
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from threading import Timer
from PIL import Image, ImageTk

def isoutlier(data, method='mean', threshold=3):
                if method == 'mean':
                    mean = np.mean(data)
                    std = np.std(data)
                    outliers = abs(data - mean) > threshold * std
                elif method == 'median':
                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    outliers = abs(data - median) > threshold * mad
                elif method == 'quartiles':
                    q1, q3 = np.percentile(data, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = (data < lower_bound) | (data > upper_bound)
                else:
                    raise ValueError('Unsupported method')
                
                return outliers


        
def run_simulation(tstart, tend, dt, Tc, plot_options, files):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.integrate import trapz
    from scipy.io import loadmat
    from scipy.spatial import KDTree
    from PIL import Image, ImageTk
    from tkinter import PhotoImage
    
    
    global TAWSS_plot, OSI_plot, RRT_plot, transWSS_plot, ECAP_plot, HOLMES_plot, TAWSS, RRT, transWSS, OSI, X,Y,Z, ECAP, HOLMES, Z_smoothed, TAWSS_dyn_cm2, selected_checkboxes, true_values
    # X_grid, Y_grid, Z_grid,X_grid_reduced,Y_grid_reduced, Z_grid_reduced

    # Set plotting parameters
    TAWSS_plot = 'TAWSS' in plot_options
    OSI_plot = 'OSI' in plot_options
    RRT_plot = 'RRT' in plot_options
    transWSS_plot = 'transWSS' in plot_options
    ECAP_plot = 'ECAP' in plot_options
    HOLMES_plot = 'HOLMES' in plot_options

    
    selected_checkboxes = {
            'tab-tawss':'TAWSS' in plot_options,
            'tab-osi': 'OSI' in plot_options,
            'tab-rrt': 'RRT' in plot_options,
            'tab-transwss':'transWSS' in plot_options,
            'tab-ecap': 'ECAP' in plot_options,
            'tab-holmes': 'HOLMES' in plot_options
        }
    
    # Load Data
    selected_files = sorted(files)

    # Number of files selected

    num_files = len(selected_files)
    print(num_files)

    # Check if the length of the time vector matches the number of files
    expected_time_steps = round((tend - tstart) / dt)
    if expected_time_steps != num_files:
        raise ValueError(f"Number of time steps ({expected_time_steps}) doesn''t match number of files ({num_files}). Check again.")

    # Separate counter for loop later
    time_point_counter = 1
    # Initial read of a file to figure out preallocation values needed
    temp = np.loadtxt(selected_files[0], skiprows=1, delimiter=',')
    # temp = temp['data']
    node_num, var_num = temp.shape
    data = np.zeros((node_num, var_num, num_files))

    progress_window= tk.Toplevel(root)
    progress_window.title("Progress")
    center_window(progress_window, width=300, height=50)
    # progress_window.geometry("300x50")

    progress_label= tk.Label(progress_window, text=" ")
    progress_label.pack()
    root.update()

    # Loop through all files and load into a 3D matrix
    for i in range(num_files):
        try:
            # print(f'File {i+1} out of {num_files} imported.')
            data_temp = np.loadtxt(selected_files[i],  skiprows=1, delimiter=',')
            progress_label.config(text=f'File {i+1} imported from range({num_files})')
            progress_window.update()
            #data_temp = data_temp['data']
            data[:, :, time_point_counter-1] = data_temp
            time_point_counter += 1
        except:
            continue

    progress_window.destroy()
    
    # Extract Position Data
    X = 1000 * data[:, 1, 0]
    Y = 1000 * data[:, 2, 0]
    Z = 1000 * data[:, 3, 0]
 
    ptCloud = KDTree(np.column_stack((X, Y, Z)))
    _, indices = ptCloud.query(np.column_stack((X, Y, Z)), k=6)
    normals = np.zeros((node_num, 3))
    for i in range(node_num):
        points = np.column_stack((X[indices[i]], Y[indices[i]], Z[indices[i]]))
        centroid = np.mean(points, axis=0)
        cov_matrix = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        if np.dot(normal, centroid) < 0:
            normal = -normal
        normals[i] = normal

    # Calculate TAWSS, OSI, RRT, transWSS, ECAP, and HOLMES
    TAWSS = np.zeros(node_num)
    OSI = np.zeros(node_num)
    RRT = np.zeros(node_num)
    transWSS = np.zeros(node_num)
    ECAP = np.zeros(node_num)
    HOLMES = np.zeros(node_num)
    WSS_Mag = np.zeros(num_files)
    WSS_X = np.zeros(num_files)
    WSS_Y = np.zeros(num_files)
    WSS_Z = np.zeros(num_files)

    t = np.arange(tstart + dt, tend + dt, dt)  # Create vector of time data

    time_point_counter = 1
    progress_window = tk.Toplevel()
    progress_window.title('Processing Nodes')
    center_window(progress_window, width=300, height=100)
    # progress_window.geometry('300x100')

    progress_label = tk.Label(progress_window, text="Processing nodes...")
    progress_label.pack()

    progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress_bar.pack()
    progress_bar['maximum'] = node_num
    progress_window.update()
    # Loop through each node and calculate the parameters
    for j in range(node_num):  # All nodes
        # print(f'Node {j+1} out of {node_num} processed.')
        progress_bar['value'] = j + 1
        progress_window.update()
        for i in range(num_files):  # All timesteps
            try:
                # Grab the variable values temporarily
                WSS_Mag_temp = data[j, 4, i]
                WSS_X_temp = data[j, 5, i]
                WSS_Y_temp = data[j, 6, i]
                WSS_Z_temp = data[j, 7, i]

                # Add values to separate variable for later
                WSS_Mag[time_point_counter-1] = WSS_Mag_temp
                WSS_X[time_point_counter-1] = WSS_X_temp
                WSS_Y[time_point_counter-1] = WSS_Y_temp
                WSS_Z[time_point_counter-1] = WSS_Z_temp

                # Increment time point counter
                time_point_counter += 1
            except Exception as e:
                print(f"Error processing data for node {j+1}, file {i+1}: {e}")
                # Continue to next loop iteration if data can't be parsed
                continue

        
        try:
        # Calculate TAWSS using trapz to approximate the integral
            TAWSS[j] = (1 / Tc) * trapz(WSS_Mag, t)
            
            # Calculate OSI using the trapz function
            top = abs(trapz(WSS_X + WSS_Y + WSS_Z, t))
            OSI[j] = 0.5 * (1 - top / trapz( np.sqrt(WSS_X**2 + WSS_Y**2 + WSS_Z**2),t))

            # Calculate RRT using TAWSS and OSI
            RRT[j] = 1 / (TAWSS[j] * (1 - 2 * OSI[j]))
            
            # Calculate transWSS
            top = np.array([trapz(WSS_X,t), trapz(WSS_Y, t), trapz(WSS_Z,t)])
            bot = abs(trapz(WSS_X + WSS_Y + WSS_Z, t))
            inner = np.cross(normals[j,:], top/ bot)
            outer = np.dot(np.column_stack((WSS_X, WSS_Y, WSS_Z)), inner)
            transWSS[j] = (1 / Tc) * trapz(abs(outer), t)
            
            # Check for NaN values in transWSS calculation and handle gracefully
        
            
    
            #calculate ECAP
            ECAP[j] = abs(OSI[j] / TAWSS[j])
            
            # Calculate HOLMES
            HOLMES[j] = TAWSS[j] * (0.5 - OSI[j])
            
        except Exception as e:
            print(f"Error calculating parameters for node {j+1}: {e}")
            continue

        # Reset WSS parsed variables for next node
        WSS_Mag = np.zeros(num_files)
        WSS_X = np.zeros(num_files)
        WSS_Y = np.zeros(num_files)
        WSS_Z = np.zeros(num_files)

        # Reset time point counter
        time_point_counter = 1
    progress_window.destroy()
    open_plot_window()

def plot_TAWSS(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> go.Figure:
    TAWSS_dyn_cm2 = TAWSS * 10  # Example calculation
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=TAWSS_dyn_cm2,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='TAWSS (dynes/cm^2)'
            )
        )
    )])

    camera = fig['layout']['scene']['camera']

    fig.update_layout(
        title='Time-Averaged Wall Shear Stress (TAWSS)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.camera': camera,
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'}]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.camera': camera,
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )

    return fig



def plot_OSI(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> go.Figure:
    
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=OSI+0.05,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='OSI'
            )
        )
    )])

    fig.update_layout(
        title='Oscillatory Shear Index (OSI)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margin as needed
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'}]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )

    return fig




def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")
    


def plot_RRT(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> go.Figure:
    RRT_Outlier_Found = np.sum(isoutlier(RRT,'mean'))

    # If found log the data for better visuals, otherwise leave as normal units of (1/Pa)
    if RRT_Outlier_Found > 0:
        C = np.log(RRT)
    else:
        C = RRT

    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=C,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='RRT'
            )
        )
    )])

    fig.update_layout(
        title='Relative Residence Time (RRT)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margin as needed
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'}]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )

    return fig



def plot_transWSS(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> Figure:
    transWSS_Outlier_Found = np.sum(np.abs(transWSS - np.mean(transWSS)) > 3 * np.std(transWSS))

    if transWSS_Outlier_Found > 0:
        C = np.log(10 * transWSS)
    else:
        C = transWSS * 10
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=C,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='TRANSWSS'
            )
        )
    )])

    fig.update_layout(
        title='Transverse Wall Shear Stress (transWSS)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margin as needed
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'
                                }]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )

    return fig

def plot_ECAP(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> Figure:
    fig = plt.figure()
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=ECAP,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='ECAP'
            )
        )
    )])

    fig.update_layout(
        title='Endothelial Cell Activation Potential (ECAP)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margin as needed
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'}]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )
    return fig


def plot_HOLMES(x_axis_label='X (mm)', y_axis_label='Y (mm)', z_axis_label='Z (mm)', 
               font_size=12, font_family='Arial') -> Figure:
# if HOLMES_plot:
    fig = plt.figure()
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=10,
            color=HOLMES,
            colorscale='Jet',
            opacity=1.0,
            colorbar=dict(
                title='HOLMES'
            )
        )
    )])


    fig.update_layout(
        title='Hemodynamic Oscillatory Lagrangian Metric of Endothelial Stress (HOLMES)',
        scene=dict(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            zaxis_title=z_axis_label,
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            yaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            zaxis=dict(
                title_font=dict(size=font_size, family=font_family),
                visible=True,
            ),
            
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margin as needed
        hovermode='closest',
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(label='Hide Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'white',
                                'scene.yaxis.color': 'white',
                                'scene.zaxis.color': 'white',
                                'scene.xaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.yaxis.backgroundcolor':'rgba(255,255,255,0.8)',
                                'scene.zaxis.backgroundcolor':'rgba(255,255,255,0.8)'}]),
                    dict(label='Show Axis',
                         method='relayout',
                         args=[{'scene.xaxis.color': 'black',
                                'scene.yaxis.color': 'black',
                                'scene.zaxis.color': 'black',
                                'scene.xaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.yaxis.backgroundcolor':'rgba(232, 236, 244, 1)',
                                'scene.zaxis.backgroundcolor':'rgba(232, 236, 244, 1)'}]),
                ],
            )
        ]
    )

    return fig


# Initialize Dash app
app = dash.Dash(__name__)
# window = webview.create_window("NewWindow", "http://127.0.0.1:8050")
app.css.append_css({"external_url": "/assets/styless.css"})

# Define app layout
app.layout = html.Div([
    html.Div(id="main", children=[
        html.Div(
    style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'},
    children=[
        html.Div(style={'display': 'flex', 'align-items': 'center'}, children=[
            html.Img(src="/assets/HaemCal.png", style={'width': '50px', 'height': 'auto'}),
            html.Label('HaemCal', style={'font-weight': 'bold', 'color': 'brown', 'font-size': '40px', 'margin-left': '10px'}),
        ]),
        html.A(href='https://sites.google.com/view/biomechanics-research-lab', target='_blank', children=html.Img(src='/assets/C.png', style={'width': '50px', 'height': 'auto', 'margin':'15px'}))
    ]), 
        html.Span("☰", style={'font-size': '30px', 'cursor': 'pointer', 'position': 'absolute', 'background-color':'rgb(60,20,164)', 'margin-top':'5px', 'border-radius':'5px', 'padding':'2px 5px 2px', 'color':'white'}, id="open-menu"),
        dcc.Tabs(id="tabs", value='tab-tawss', children=[]), 
        html.Div(id='tabs-content'),
    ]),
    html.Div(id="sidenav", className="sidenav", style={'width': '0px'}, children=[
        html.Div(className='sidenav-top',children=[
            html.Img(src="/assets/HaemCal.png", style={'width':'50px', 'height':'auto'}),
            html.Label('HaemCal', style={'font-weight':'bold','color':'brown','font-size':'40px'}),
            html.Span("×", style={'font-size': '30px', 'cursor': 'pointer', 'position': 'absolute', 'top': '10px', 'right': '25px'}, id="close-menu"),
        ]), 
        html.Div(style={'margin-bottom':'10px', 'margin-left':'10px'}, children=[
            html.Label('X Axis Label:'),
            dcc.Input(id='x-axis-label', value='X (mm)', type='text'),
        ]),
        html.Div(style={'margin-bottom':'10px', 'margin-left':'10px'}, children=[
            html.Label('Y Axis Label:'),
            dcc.Input(id='y-axis-label', value='Y (mm)', type='text'),
        ]),
        html.Div(style={'margin-bottom':'10px', 'margin-left':'10px'}, children=[
            html.Label('Z Axis Label:'),
            dcc.Input(id='z-axis-label', value='Z (mm)', type='text'),
        ]),
        html.Div(style={'margin-bottom':'10px', 'margin-left':'10px'}, children=[
            html.Label('Font Size for Axes Labels:'),
            dcc.Slider(
                id='font-size-slider',
                min=10,
                max=30,
                step=1,
                value=12,
                marks={i: str(i) for i in range(10, 31, 5)},
            ),
        ]),
        html.Div(style={'margin-bottom':'10px', 'margin-left':'10px'}, children=[
            html.Label('Font Family for Axes Labels:'),
            dcc.Dropdown(
                id='font-family-dropdown',
                options=[
                    {'label': 'Arial', 'value': 'Arial'},
                    {'label': 'Times New Roman', 'value': 'Times New Roman'},
                    {'label': 'Courier New', 'value': 'Courier New'},
                    {'label': 'Verdana', 'value': 'Verdana'},
                    {'label': 'Calibri', 'value': 'Calibri'},
                    {'label': 'Tahoma', 'value': 'Tahome'},
                    {'label': 'Georgia', 'value': 'Georgia'},
                    {'label': 'Century Gothic', 'value': 'Century Gothic'},
                    {'label': 'Comic Sans MS', 'value': 'Comic Sans MS'},
                    {'label': 'Garamond', 'value': 'Garamond'}
                ],
                value='Arial'
            ),
        ]),
    ]),
])

# Callback to toggle the side menu
@app.callback(
    Output("sidenav", "style"),
    [Input("open-menu", "n_clicks"), Input("close-menu", "n_clicks")],
    [State("sidenav", "style")]
)
def toggle_sidenav(open_clicks, close_clicks, sidenav_style):
    if not sidenav_style:
        sidenav_style = {'width': '0px'}  # Initial width set to 0px

    ctx = dash.callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "open-menu" and (not sidenav_style or sidenav_style['width'] == '0px'):
            return {'width': '250px', 'background-color': 'white'}  # Open sidebar
        elif button_id == "close-menu":
            return {'width': '0px'}  # Close sidebar

    return sidenav_style

# Dummy callback for updating tabs (replace with your actual logic)
@app.callback(
    Output('tabs', 'children'),
    [Input('x-axis-label', 'value')]
)
def update_tabs(_):
    tabs = []
    if selected_checkboxes['tab-tawss']:
        tabs.append(dcc.Tab(label='TAWSS', value='tab-tawss'))
    if selected_checkboxes['tab-osi']:
        tabs.append(dcc.Tab(label='OSI', value='tab-osi'))
    if selected_checkboxes['tab-rrt']:
        tabs.append(dcc.Tab(label='RRT', value='tab-rrt'))
    if selected_checkboxes['tab-transwss']:
        tabs.append(dcc.Tab(label='transWSS', value='tab-transwss'))
    if selected_checkboxes['tab-ecap']:
        tabs.append(dcc.Tab(label='ECAP', value='tab-ecap'))
    if selected_checkboxes['tab-holmes']:
        tabs.append(dcc.Tab(label='HOLMES', value='tab-holmes'))
    return tabs

# Callback to update the content of tabs based on selected tab
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('x-axis-label', 'value'),
     Input('y-axis-label', 'value'),
     Input('z-axis-label', 'value'),
     Input('font-size-slider', 'value'),
     Input('font-family-dropdown', 'value')]
)
def render_content(tab, x_axis_label, y_axis_label, z_axis_label, font_size, font_family):
    graph_style = {
        'width': '100%',
        'height': '80vh'  
    }
    if tab == 'tab-tawss':
        return html.Div([
            dcc.Graph(id='graph-tawss', figure=plot_TAWSS(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    elif tab == 'tab-osi':
        return html.Div([
            dcc.Graph(id='graph-osi', figure=plot_OSI(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    elif tab == 'tab-rrt':
        return html.Div([
            dcc.Graph(id='graph-rrt', figure=plot_RRT(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    elif tab == 'tab-transwss':
        return html.Div([
            dcc.Graph(id='graph-transwss', figure=plot_transWSS(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    elif tab == 'tab-ecap':
        return html.Div([
            dcc.Graph(id='graph-ecap', figure=plot_ECAP(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    elif tab == 'tab-holmes':
        return html.Div([
            dcc.Graph(id='graph-holmes', figure=plot_HOLMES(x_axis_label, y_axis_label, z_axis_label, font_size, font_family), style=graph_style)
        ])
    
@app.callback(
    [Output('graph-tawss', 'figure'),
     Output('graph-osi', 'figure'),
     Output('graph-rrt', 'figure'),
     Output('graph-transwss', 'figure'),
     Output('graph-ecap', 'figure'),
     Output('graph-holmes', 'figure')],
    [Input('x-axis-label', 'value'),
     Input('y-axis-label', 'value'),
     Input('z-axis-label', 'value'),
     Input('font-size-slider', 'value'),
     Input('font-family-dropdown', 'value'),
     ],
)

def update_graph(x_axis_label, y_axis_label, z_axis_label, font_size, font_family):
    
    fig_tawss = plot_TAWSS(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    fig_osi = plot_OSI(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    fig_rrt = plot_RRT(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    fig_transwss = plot_transWSS(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    fig_ecap = plot_ECAP(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    fig_holmes = plot_HOLMES(x_axis_label, y_axis_label, z_axis_label, font_size, font_family)
    
    return fig_tawss, fig_osi, fig_rrt, fig_transwss, fig_ecap, fig_holmes


def select_files():
    files = filedialog.askopenfilenames(filetypes=[("All files", ".*")])
    files_entry.delete(0, tk.END)
    files_entry.insert(0, ', '.join(files))

def submit():
    try:
        tstart = float(tstart_entry.get())
        tend = float(tend_entry.get())
        dt = float(dt_entry.get())
        Tc = float(Tc_entry.get())
        
        # Validate plot parameters selection
        plot_options = [option for option, var in plot_vars.items() if var.get()]
        if not plot_options:
            raise ValueError("Please select at least one plot parameter.")
        
        # Validate files selection
        files = files_entry.get().split(', ')
        if not files:
            raise ValueError("No files selected.")
        
        run_simulation(tstart, tend, dt, Tc, plot_options, files)
        
    except ValueError as e:
        messagebox.showerror("Input error", str(e))


def open_plot_window():
    Timer(1, open_browser).start()
    root.withdraw()
    root.destroy()
    app.run_server(debug=False)
    
    # Sample data for demonstration
    if TAWSS_plot:
        fig = plot_TAWSS()
        
    if OSI_plot:
        fig_osi = plot_OSI()
       
    if RRT_plot:
        fig_rrt = plot_RRT()

    if transWSS_plot:
        fig_transwss = plot_transWSS()
        
    if ECAP_plot:
        fig_ecap = plot_ECAP()
        

    if HOLMES_plot:
        fig_holmes = plot_HOLMES()

    


root = tk.Tk()
root.title("Hemodynamics Simulator")


script_dir= os.path.dirname(__file__)
image_path = os.path.join(script_dir, "assets", "HaemCal.ico" )
root.iconbitmap(image_path)

logo_path=os.path.join(script_dir, "Images", "labLogo.png" )
iit_logo_path = os.path.join(script_dir, "Images","Official_Logo_of_IIT(BHU),Varanasi,India,2013.png")
def center_window(window, width, height):
    # get screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    window.geometry('%dx%d+%d+%d' % (width, height, x, y))

# Center the window on startup
center_window(root, width=520, height=600)

def on_enter_press(event):
    focus_next_widget(event.widget)


def focus_next_widget(widget):
    widget.tk_focusNext().focus()


haemCal_logo = os.path.join(script_dir, "assets", "HaemCal.png" )
try:
    image = Image.open(haemCal_logo)
    resized_image = image.resize((66,66))
    logoHC = ImageTk.PhotoImage(resized_image)
    canvas = tk.Canvas(root, width=66, height=66)
    canvas.create_image(0, 0, anchor='nw', image=logoHC)
    canvas.grid(row=0, column=0, padx=15, pady=10, sticky='nw')
except tk.TclError:
    messagebox.showerror("Error", "Image file not found.")

tk.Label(root, text="HaemCal", font=('Times New Roman', 30), fg='maroon').grid(row=0, column=1, pady=10, sticky='nw')

tk.Label(root, text="Start Time:", font=('Calibri', 10)).grid(row=2, column=0, padx=10, pady=2, sticky='w')
tstart_entry = tk.Entry(root)
tstart_entry.grid(row=2, column=1, padx=10, pady=2, sticky='w')
tstart_entry.bind('<Return>', on_enter_press)

tk.Label(root, text="End Time:", font=('Calibri', 10)).grid(row=3, column=0, sticky='w', padx=10, pady=2)
tend_entry = tk.Entry(root)
tend_entry.grid(row=3, column=1, padx=10, pady=2, sticky='w')
tend_entry.bind('<Return>', on_enter_press)

tk.Label(root, text="Time Step:", font=('Calibri', 10)).grid(row=4, column=0, sticky='w', padx=10, pady=2)
dt_entry = tk.Entry(root)
dt_entry.grid(row=4, column=1, padx=10, pady=2, sticky='w')
dt_entry.bind('<Return>', on_enter_press)


tk.Label(root, text="Cardiac Cycle Time:", font=('Calibri', 10)).grid(row=5, column=0, sticky='w', padx=10, pady=(2, 10))
Tc_entry = tk.Entry(root)
Tc_entry.grid(row=5, column=1, padx=10, pady=(2, 10), sticky='w')
Tc_entry.grid(row=5, column=1, padx=10, pady=(2, 10), sticky='w')

tk.Label(root, text="Select Parameters for plotting:", font=('Calibri', 13)).grid(row=6, column=0, padx=10, pady=(5, 0), columnspan=2, sticky='w')

def select_all():
    all_selected = all(var.get() for var in plot_vars.values())
    
    if all_selected:
        # Deselect all options
        for var in plot_vars.values():
            var.set(False)
    else:
        # Select all options
        for var in plot_vars.values():
            var.set(True)

select_all_button = tk.Button(root, text="Select All", command=select_all)
select_all_button.grid(row=7, column=0, columnspan=2, sticky='w', padx=10, pady=(0, 5))

plot_vars = {}
plot_options = ["TAWSS", "OSI", "RRT", "transWSS", "ECAP", "HOLMES"]
for i, option in enumerate(plot_options):
    var = tk.BooleanVar()
    plot_vars[option] = var
    cb = tk.Checkbutton(root, text=option, variable=var)
    cb.grid(row=8+i, column=0, columnspan=2, sticky='w', padx=10, pady=0)

try:
    logo1 = tk.PhotoImage(file=iit_logo_path)
    canvas = tk.Canvas(root, width=145, height=logo1.height())
    canvas.create_image(0, 0, anchor='nw', image=logo1)
    canvas.grid(row=17, column=0, padx=20, sticky='nw')
except tk.TclError:
    messagebox.showerror("Error", "Image file not found.")

try:
    logo2 = tk.PhotoImage(file=logo_path)
    canvas = tk.Canvas(root, width=66, height=66)
    canvas.create_image(0, 0, anchor='nw', image=logo2)
    canvas.grid(row=17, column=1, padx=20, sticky='ne')
except tk.TclError:
    messagebox.showerror("Error", "Image file not found.")


tk.Label(root, text="Biomechanics Research Lab, SBME, IIT-BHU, Varanasi").grid(row=18, column=0, columnspan=2)
tk.Button(root, text="Select Files", command=select_files).grid(row=14, column=0, padx=10, pady=5, sticky='nw')
files_entry = tk.Entry(root, width=50)
files_entry.grid(row=14, column=1, padx=10, pady=5, sticky='w')

tk.Button(root, text="Run Simulation", command=submit).grid(row=16, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()