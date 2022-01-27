# Put this in the InputGroup where InputGroupText exists currently.
dbc.Select(
  id='units',
  options=[{'label': unit, 'id': unit} for unit in ['meters','feet']],
  value='meters'
)


@app.callback(
  Output('slider', 'min'),
  Output('slider', 'max'),
  Output('slider', 'marks'),
  Output('slider', 'value'),
  Input('units', 'value')
)
def update_slider(units):
  pass


def create_slider(distance_series, units):
  d_min = distance_series.min()
  d_max = distance_series.max()

  return dcc.RangeSlider(
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
  )