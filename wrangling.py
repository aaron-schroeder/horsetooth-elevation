"""wrangling.py"""
import heartandsole

act = heartandsole.Activity.from_gpx('data/Horsetooth Half Marathon.gpx')

# If distances are not included in the file, calculate them from lat/lon.
if not act.has_streams('distance'):
  act.distance.records_from_position(inplace=True)

# TODO: Determine if I want to save any grade values in-file.
# if act.has_streams('elevation'):
#   import numpy as np  # to perform a central diff method, not simple differences between pts.
  
#   # act.records['grade'] = 100.0 * act.records['elevation'].diff() / act.records['distance'].diff()
#   act.records['grade'] = 100.0 * act.records.xyz.z_smooth_distance().diff() / act.records['distance'].diff()
#   # act.records['grade'] = 100.0 * np.gradient(act.records.xyz.z_smooth_distance(), act.records['distance'])
#   # act.records['grade'] = 100.0 * np.gradient(act.records['elevation'], act.records['distance'])

# Save DF as a CSV file
act.records.to_csv('data/horsetooth.csv')