# Changelog for major revisions of the sled processing code

## Revision: 0.99, 8/30/2022, M. Gallagher
The initial version of this code was improvements/theft of the ASFS-AMOS flux buoy "real-time processing" code. Bug fixes and revisions have been made (and will be backported). Version 0.99 produces "raw" uncurated level1 netcdf files for the fast and the slow data, as well as an "experimental" turbulent flux data product on 10, 30, and 60 minute timesteps. The code is optimized with the intent to be run on individual days, (multi-threaded internally for turbulent calculations, not across days like the MOSAiC)  and as such is slower than it could be when processing the full dataset. It will likely remain this way so that "real-time" data production sled-side is possible. 
