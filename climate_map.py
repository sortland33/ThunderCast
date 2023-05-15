import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys, os
import numpy as np
import cartopy.io.shapereader as shpreader
import matplotlib.patches as mpatches

northwest = ['Washington', 'Oregon', 'Idaho']
west = ['California', 'Nevada']
west_north_central = ['Montana', 'Nebraska', 'North Dakota', 'South Dakota',
                      'Wyoming']  # northern rockies and plains
southwest = ['Arizona', 'Colorado', 'New Mexico', 'Utah']
east_north_central = ['Minnesota', 'Michigan', 'Iowa', 'Wisconsin']  # upper midwest
south = ['Arkansas', 'Kansas', 'Louisiana', 'Mississippi', 'Oklahoma', 'Texas']
southeast = ['Alabama', 'Florida', 'Georgia', 'North Carolina', 'South Carolina', 'Virginia']
central = ['Illinois', 'Indiana', 'Kentucky', 'Missouri', 'Ohio', 'Tennessee', 'West Virginia']  # Ohio Valley
northeast = ['Connecticut', 'Delaware', 'Maine', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York',
             'Pennsylvania', 'Rhode Island', 'Vermont', 'Maryland']

#state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')
shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_1_states_provinces_lakes')
reader = shpreader.Reader(shpfilename)
states = reader.records()
state_geom = reader.geometries()
projPC = ccrs.PlateCarree()
lonW = -125
lonE = -66
latS = 24
latN = 50
cLat = (latN + latS) / 2
cLon = (lonW + lonE) / 2
proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
fig = plt.figure(figsize=(14, 10))
ax = plt.subplot(1, 1, 1, projection=proj)
#ax.set_title('Stereographic')
# gl = ax.gridlines(
#     draw_labels=False, linewidth=2, color='gray', alpha=0.5, linestyle='--'
# )
ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
west_patch = mpatches.Patch(color='tab:brown', label='West')
west_north_central_patch = mpatches.Patch(color='tab:orange', label='West North Central')
southwest_patch = mpatches.Patch(color='tab:green', label='Southwest')
northwest_patch = mpatches.Patch(color='tab:red', label='Northwest')
east_north_central_patch = mpatches.Patch(color='tab:purple', label='East North Central')
south_patch = mpatches.Patch(color='tab:blue', label='South')
southeast_patch = mpatches.Patch(color='tab:pink', label='Southeast')
central_patch = mpatches.Patch(color='tab:olive', label='Central')
northeast_patch = mpatches.Patch(color='tab:cyan', label='Northeast')
ax.legend(handles=[west_patch, west_north_central_patch, southwest_patch, northwest_patch, east_north_central_patch,
                   south_patch, southeast_patch, central_patch, northeast_patch], loc='lower right', fontsize=12)

edgecolor = 'black'
for i, state in enumerate(states):
    if state.attributes['name'] in west:
        facecolor = 'tab:brown'
    elif state.attributes['name'] in west_north_central:
        facecolor = 'tab:orange'
    elif state.attributes['name'] in southwest:
        facecolor = 'tab:green'
    elif state.attributes['name'] in northwest:
        facecolor = 'tab:red'
    elif state.attributes['name'] in east_north_central:
        facecolor = 'tab:purple'
    elif state.attributes['name'] in south:
        facecolor = 'tab:blue'
    elif state.attributes['name'] in southeast:
        facecolor = 'tab:pink'
    elif state.attributes['name'] in central:
        facecolor = 'tab:olive'
    elif state.attributes['name'] in northeast:
        facecolor = 'tab:cyan'
    else:
        facecolor = 'tab:gray'
    ax.add_geometries([state.geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor=edgecolor)
figure_path = '/ships19/grain/convective_init/paper_plots/climate_regions.png'
plt.savefig(figure_path, bbox_inches='tight', dpi=200)