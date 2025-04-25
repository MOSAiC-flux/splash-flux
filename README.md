# SPLASH ASFS processing code

This code currently produces SPLASH processed netcdf files and flux calculations, it is based on significant efforts in the MOSAiC campaign and the AMOS buoy project. These codes produce three versions of files:

## "level 1 raw/realtime product" 

The level 1 processing creates two type of products. A "realtime" flux product that allows for the scientific study of preliminary results. This derived realtime product is not evaluated or processed in a meaningfully rich way and any conclusions from these data should be carefully considered. Research should not be done using this product beyond quicklooks while data is being gathered. 

## "level 2 quality controlled" 

Considerable quality control and effort goes into the processing and combination of data for the level 2 product. The files contain flagging for sensors/variables to indicate the important considerations to be made in using the data. Poor quality data is flagged or thrown out and each processing is applied to datastreams and derivations to ensure the resultings product best represents the measured parameter of interest. 

Level 2 files are updated as effort is applied to improve the data. Versions of these data may be frequently revised/updated as the project progresses. 

## "level 3 archive product" 

Level 3 data is a streamlined version of the data from the finalized level 2 files that drops variables that aren't considered of great import for an archive-quality research product. Data that was flagged in the level2 files as either bad or engineering are also dropped. Any data in these files are of research grade. 