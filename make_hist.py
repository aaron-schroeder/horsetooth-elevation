import plotly.express as px

from app import read_csv

df = read_csv('data/horsetooth.csv')

med = df['distance'].diff().median()
print(f'median = {med:.1f} meters')

# Make a histogram of the point-to-point spacing
hist = px.histogram(df.diff(), x='distance', nbins=20)
hist.add_vline(x=med, line_width=2, line_dash='dash', line_color='green')

# hist.show()
hist.write_image('images/hist.jpeg')