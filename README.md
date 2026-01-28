# miniseed_images 

## [geophone_timing](miniseed_images/geophone_timing.py)

This script is for checking start and end time of each channel of a single miniseed file, as well as checking other metadata such as sampling rate, number of data points per channel, missing data points etc. 

## [plot_ms_trace.py](miniseed_images/plot_ms_trace.py)

This script is for plotting merging or appending the missing data segment from the duplicated channels in a single miniseed file. 

## [read_ms_to_create_images.py](miniseed_images/read_ms_to_create_images.py)

This script is to generate images for ML processing. Using obspy trace merger to append the leftover data segments from the duplicated channels of the same station is implemented to ensure accurate data handling. 

# Cautions

** The images created before merging the duplicated channels together should not make a difference to the ML seismicity results. In fact, it is highly unlikely that the leftover segment would contain events. This is currently a work-around and a temporary solution. However, if the original miniseed processing logic is corrected, then such work-around is no longer needed. **

** The timestamps of each data point from each channel are indeed aligned so using old logic (without extra merging of the duplicated channels) will not have flow-on effects on seismic event location. **