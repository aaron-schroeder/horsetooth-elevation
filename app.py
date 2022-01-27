"""Display data from a csv Activity file in an interactive dashboard."""
import math

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

SAMPLE_LEN = 100.0


def create_dash_app():

  app = Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
      # 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.8.3/plotly-mapbox.js',
      'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.8.3/plotly-mapbox.min.js',
    ]
  )

  df = read_csv('data/horsetooth.csv')

  d_min = df['distance'].min()
  d_max = df['distance'].max()

  app.layout = dbc.Container(
    [
      dcc.Loading(
        id='loading-1',
        type='default',
        children=html.Div(
          id='loading-output-1',
          children=[
            dcc.Graph(
              id='figure', 
              figure=create_fig(df),
            ),
          ]
        )
      ),
      dcc.Store( # data streams
        id='activity-data',
        data=df.to_dict('records'),
      ),
      dbc.Row(
        [
          dbc.Col(
            dcc.RangeSlider(
              id='slider',
              # min=0,
              # max=len(df) - 1,
              min=d_min,
              max=d_max,
              # step=1,
              # step=None,
              marks={
                1000 * i: f'{i}k' for i in range(math.ceil(d_max / 1000))
              },
              # value=[0, len(df) - 1],
              value=[d_min, d_max],
              # value=[df['distance'][0], df['distance'].iloc[-1]],
              allowCross=False,
              tooltip=dict(
                placement='bottom',
                always_visible=False
              ),
              # style={'padding': '0px 0px 25px'},
              className='px-1',
            ),
            width=12,
            className='mb-3',
          ),
        ],
        justify='center'
      ),
      dbc.Row([
        dbc.Col(
          [
            dbc.InputGroup([
              dbc.InputGroupText('Sample elevation every'),
              dbc.Input(
                id='len-sample',
                type='number', min=5, max=100,
                step=5, 
                value=100,
              ),
              dbc.InputGroupText('meters'),
            ]),
            dbc.FormText('Choose a value between 5 and 100.'),
          ],
          # width=12,
          md=6,
        ),
      ]),
      dcc.Loading(
        id='loading-2',
        type='default',
        children=html.Div(
          id='loading-output-2',
          children=[
            html.Hr(),
            html.H2('Hill Statistics'),
            html.Div(id='stats')
          ],
        ),
      ),
    ],
    id='dash-container',
    # fluid=True,
  )

  # Initialize callbacks
  @app.callback(
    Output('figure', 'figure'),
    Output('stats', 'children'),
    Input('slider', 'value'),
    Input('len-sample', 'value'),
    State('activity-data', 'data'),
    # State('figure', 'figure'),
  )
  def update_fig_and_stats(slider_values, len_sample, record_data):
    """Update plot and statistics to desired range and sample interval."""

    if record_data is None:
      raise PreventUpdate

    len_sample = len_sample or SAMPLE_LEN

    df = pd.DataFrame.from_records(record_data)

    df_sub = sample_dist(df, slider_values[0], slider_values[1], len_sample)

    fig = create_fig(df, df_sub)

    stats_children = create_stats_table(df_sub)

    # FIX this. The highlighted points no longer make sense.
    # fig.update_traces(selectedpoints=df_sub.index)

    return fig, stats_children

  return app


def create_fig(df, df_sub=None):
  """Generate a plotly figure of an elevation profile.

  Creates a plotly figure for use as `dcc.Graph.figure` representing
  the elevation profile contained in the input DataFrame. The first
  trace represents the raw data. If ``df_sub`` is passed as an input,
  a second trace is added with the sub-sampled elevation profile data,
  including a filled area between each pair of points representing the
  grade between them.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    df_sub (pd.DataFrame): Optional. A DataFrame representing a subset
      of the data in ``df``, re-sampled at even distance intervals.

  Returns:
    plotly.graph_objects.Figure: figure to be used with `dcc.Graph`
  """
  fig = go.Figure(layout=dict(
    xaxis=dict(
      range=[df['distance'].min(), df['distance'].max()],
      showticklabels=False
    ),
    yaxis=dict(
      range=[0.9 * df['elevation'].min(), 1.1 * df['elevation'].max()],
      showticklabels=False
    ),
    margin=dict(
      b=0,
      # t=0,
      r=0,
      l=0,
    ),
    showlegend=False,
  ))

  # ----------------------------------------------------------------------
  # Identify grade with filled regions

  if df_sub is not None:

    x = df_sub['distance']
    y = df_sub['elevation']

    dydx = 100 * np.diff(y) / np.diff(x)

    # colorscale = 'Picnic'
    colorscale = 'Tropic'

    # Add a dummy trace so the colorbar will appear on the plot.
    fig.add_trace(go.Scatter(
      x=0.5 * (x[:-1] + x[1:]),
      y=[0.5 * fig.layout.yaxis.range[0] for i in range(len(x))],
      mode='markers',
      text=dydx,
      hoverinfo='skip',
      marker=dict(
        size=None,
        color=dydx,
        # cmid=0.0,
        cmin=-25.0,
        cmax=25.0,
        colorbar=dict(
          title='Grade',
          orientation='h',
        ),
        colorscale=colorscale,
      ),
      showlegend=False,
      # visible='legendonly',
    ))
    
    from plotly.colors import sample_colorscale
    # cmax_abs = max(abs(dydx))
    cmax_abs = 25
    fill_colors = sample_colorscale(colorscale, 0.5*(dydx/cmax_abs + 1), colortype='rgb')

    for x_pair, y_pair, fill_color, dydx_i in zip(zip(x, x[1:]), zip(y, y[1:]), fill_colors, dydx):

      fig.add_trace(go.Scatter(
        x=[x_pair[0], x_pair[0], x_pair[1], x_pair[1]],
        y=[0, y_pair[0], y_pair[1], 0],
        fill='toself',
        # x=x_pair,
        # y=y_pair,
        # fill='tozeroy',
        name=f'Grade: {dydx_i:.1f}%',
        # x=x_pair,
        # y=y_pair,
        mode='lines',
        # mode='lines+markers',
        # mode='markers',
        # mode=None,
        marker=dict(
          size=2,
        ),
        line=dict(
          # color='gray',
          width=0,
        ),
        fillcolor=fill_color,
        hoveron='fills', # select where hover is active
        showlegend=False,
      ))

    fig.add_trace(go.Scatter(
      x=x,
      y=y,
      name='Re-sampled data',
      # mode='lines',
      # mode='lines+markers',
      mode='markers',
      # mode=None,
      marker=dict(
        size=2,
        color='gray',
      ),
      hovertemplate='%{y:.1f} m at %{x:.1f} m',
    ))

  # ----------------------------------------------------------------------


  # Show the raw data so we can tell what we're sampling from
  fig.add_trace(go.Scatter(x=df['distance'], y=df['elevation'], name='Raw data', 
    # mode='lines+markers',
    # mode='lines',
    # line=dict(
      # color='black',
      # width=1,
    # ),
    mode='markers',
    marker=dict(
      color='black',
      size=4,
    ),
    legendrank=1,  # show at top of list in legend
    hovertemplate='%{y:.1f} m at %{x:.1f} m',
  ))

  return fig


def create_stats_table(df_sub=None):
  """Generate a table of statistics about an elevation profile.

  Creates a list for use as `dcc.Div.children` containing a single
  `dbc.Table` with statistics about the elevation profile data in
  ``df_sub``. Start and end locations, total length and elevation gain,
  and average/min/max grade.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    df_sub (pd.DataFrame): A DataFrame representing an elevation profile.
      Assumed to have ``elevation`` and ``distance`` columns, with the
      ``distance`` data points evenly spaced.

  Returns:
    list(dbc.Table): children to be used with `dcc.Div`. Just contains
    one table.
  """
  y_1 = df_sub['elevation'].iloc[-1]
  y_0 = df_sub['elevation'].iloc[0]
  x_1 = df_sub['distance'].iloc[-1]
  x_0 = df_sub['distance'].iloc[0]
  dx = x_1 - x_0
  dy = y_1 - y_0
  dydx = df_sub['elevation'].diff() / df_sub['distance'].diff()

  table_header = [
    html.Thead([
      html.Tr([
        html.Th('Hill', colSpan=2),
        # html.Th('Net elevation difference', rowSpan=2),
        html.Th(f'Grade (sampled every {dx/len(df_sub):.1f} m)', colSpan=3),
      ]),
      html.Tr([
        html.Th('Segment start'),
        # html.Th('Segment end'),
        html.Th('Segment length'), 
        # html.Th('Net elevation difference'),
        html.Th('Avg.'),
        html.Th('Max.'),
        html.Th('Min.'),
      ])
    ])
  ]

  row1 = html.Tr([
    html.Td(f"{x_0:.0f} m"),
    # html.Td(f"{x_1:.0f} m"),
    # html.Td(f"{dx:.0f} m / {dx/1609.34:.1f} mi"),
    # html.Td(f"{dy:.0f} m / {dy*5280/1609.34:.0f} ft"),
    html.Td(f"{dx:.0f} m"),
    # html.Td(f"{dy:.0f} m"),
    html.Td(f"{100*dy/dx:.1f}%"),
    html.Td(f"{100*dydx.max():.1f}%"),
    html.Td(f"{100*dydx.min():.1f}%"),      
  ])

  table_body = [html.Tbody([row1])]

  return [dbc.Table(
    table_header + table_body,
    bordered=True,
    style={'text-align': 'center'},
  )]


def read_csv(fname):
  return pd.read_csv(fname,
    index_col=[0],
    # header=[0, 1], 
    # parse_dates=[2]
  )


def sample_dist(df, bound_lo=None, bound_hi=None, len_sample=5.0):
  """Subsample elevation data in evenly-spaced distance intervals.
  
  First, an evenly-spaced array of distance values is generated,
  spanning from ``bound_lo`` (or the lowest distance value in ``df``)
  to ``bound_hi`` (or the highest distance value in ``df``). The spacing
  is as close to ``len_sample`` as possible without exceeding it.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    bound_lo (float): Optional. The desired lower bound of the returned
      sub-sampled DataFrame's ``distance`` column.
    bound_hi (float): Optional. The desired upper bound of the returned
      sub-sampled DataFrame's ``distance`` column.
    len_sample (float): The maximum desired point-to-point spacing, in
      meters. Default 5.0.
      
  Returns:
    pd.DataFrame: a subset of the input DataFrame, resampled at evenly-
    spaced distance coordinates. Contains only ``distance`` and 
    ``elevation`` columns.
  """
  # TODO: Verify bound_lo and bound_hi make sense

  bound_lo = bound_lo or df['distance'].iloc[0]  
  bound_hi = bound_hi or df['distance'].iloc[-1]

  n_sample = math.ceil(
    (bound_hi - bound_lo) / len_sample
  )

  distance_ds = np.linspace(
    bound_lo,
    bound_hi, 
    n_sample + 1
  )

  interp_fn = interp1d(df['distance'], df['elevation'], 
    kind='linear'
    # kind='quadratic'
    # kind='cubic'
  )

  return pd.DataFrame(dict(
    distance=distance_ds,
    elevation=interp_fn(distance_ds)
  ))


if __name__ == '__main__':
  app = create_dash_app()
  app.run_server(
    # debug=True
  )