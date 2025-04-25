#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
from sled_data_definitions import code_version
code_version = code_version()

# ############################################################################################
# AUTHORS:
#
#   Michael Gallagher (CIRES/NOAA)  michael.r.gallagher@noaa.gov
#   Christopher Cox (NOAA) christopher.j.cox@noaa.gov
#
#   Many sections based on Matlab code written by O. Persson (CIRES), A. Grachev (CIRES/NOAA),
#   and C. Fairall (NOAA); We thank them for their many contributions.
#
# PURPOSE:
#
# To create a uniform data product for the two ASFS, "flux statoins" deployed to SPLASH. 
# This code inherits some legacy from the MOSAiC data processing but attempts to improve on the
# naive implementations of certain aspects of the code. 
# 
# Descriptions of the three different output files and their contents:
# (look at "sled_data_definitions.py" for detailed product description)
# #########################################################################
#
# 
# HOWTO:
#
# To run this package with verbose printing over all of the data:
# nice -n 20 time python3 -u ./create_level2_product_sled.py -v -s 20211010 -e 20230301 -pd ./ramdisk/ -a asfs30 | tee sled_(date +%F).log
#
# To profile the code and see what's taking so long:
# python3 -m cProfile -s cumulative ./create_level2_product_asfs.py etc etc -v -s 20191201 -e 20191201 
#
# ###############################################################################################
#
# look at these files for documentation on netcdf vars and nomenclature
# as well as additional information on the syntax used to designate certain variables here
from sled_data_definitions import define_global_atts, define_level2_variables, define_turb_variables, define_qc_variables
from sled_data_definitions import define_level1_slow, define_level1_fast, define_10hz_variables

from qc_level2 import qc_asfs_winds, qc_stations, qc_asfs_turb_data
from get_data import get_splash_data

import functions_library as fl # includes a bunch of helper functions that we wrote

# Ephemeris
# SPA is NREL's (Ibrahim Reda's) emphemeris calculator that all those BSRN/ARM radiometer geeks use ;) 
# pvlib is NREL's photovoltaic library
from pvlib import spa 
    # .. [1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
    #    applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.

    # .. [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    #    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838,
    #    2007.

import os, inspect, argparse, time, sys, socket

global nthreads 
hostname = socket.gethostname()
if '.psd.' in hostname:
    if hostname.split('.')[0] in ['linux1024', 'linux512']:
        nthreads = 24  # the twins have 32 cores/64 threads, won't hurt if we use <30 threads
        nthreads = 12  # 
    else:
        nthreads = 90  # the new compute is hefty.... real hefty

else: nthreads = 8     # laptops don't tend to have 12  cores... yet

from multiprocessing import Process as P
from multiprocessing import Queue   as Q

# need to debug something? this makes useful pickle files in ./tests/ ... uncomment below if you want to kill threading
we_want_to_debug = True
if we_want_to_debug:

    from multiprocessing.dummy import Process as P
    from multiprocessing.dummy import Queue   as Q
    nthreads = 1
    try: from debug_functions import drop_me as dm
    except: you_dont_care=True
     
import numpy  as np
import pandas as pd
import xarray as xr

pd.options.mode.use_inf_as_na = True # no inf values anywhere

from datetime  import datetime, timedelta
from numpy     import sqrt
from scipy     import stats
from netCDF4   import Dataset, MFDataset, num2date

import warnings; warnings.filterwarnings(action='ignore') # vm python version problems, cleans output....


version_msg = '\n\n2021-2023 SPLASH sled processing code v.'+code_version[0]\
              +', last updates: '+code_version[1]+' by '+code_version[2]+'\n\n'

print('---------------------------------------------------------------------------------------------')
print(version_msg)

def main(): # the main data crunching program

    # the UNIX epoch... provides a common reference, used with base_time
    global epoch_time
    epoch_time        = datetime(1970,1,1,0,0,0) # Unix epoch, sets time integers

    global integ_time_turb_flux, rotation_flag
    integ_time_turb_flux = [10]                  # [minutes] the integration time for the turbulent flux calculation
    calc_fluxes          = True                  # if you want to run turbulent flux calculations and write files
    rotation_flag        = 2                     # Rotation method for sonic turbulence calculation: 1 = double; 2 = planar; 3 = single 

    global verboseprint  # defines a function that prints only if -v is used when running
    global printline     # prints a line out of dashes, pretty boring
    global verbose       # a useable flag to allow subroutines etc when using -v 
    global tilt_data     # some variables seem to be unavailable when others defined similarly are ...????
    
    # constants for calculations
    global nan, def_fill_int, def_fill_flt # make using nans look better
    nan = np.NaN
    def_fill_int = -9999
    def_fill_flt = -9999.0

    Rd         = 287     # gas constant for dry air
    K_offset   = 273.15  # convert C to K
    h2o_mass   = 18      # are the obvious things...
    co2_mass   = 44      # ... ever obvious?
    sb         = 5.67e-8 # stefan-boltzmann
    emis_snow  = 0.985   # snow emis assumption following Andreas, Persson, Miller, Warren and so on
    emis_grass = 0.975 # grassland https://www.sciencedirect.com/science/article/pii/S0034425719301555

    global version  # names directory where data will be written
    global lvlname  # will appear in filename
    lvlname = 'level2.0' 

    # there are two command line options that effect processing, the start and end date...
    # ... if not specified it runs over all the data. format: '20191001' AKA '%Y%m%d'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_time', metavar='str', help='beginning of processing period, Ymd syntax')
    parser.add_argument('-e', '--end_time', metavar='str', help='end  of processing period, Ymd syntax')
    parser.add_argument('-v', '--verbose', action ='count', help='print verbose log messages')
    parser.add_argument('-p', '--path', metavar='str', help='base path of data location, up to andincluding /data/, include trailing slash') 
    parser.add_argument('-a', '--station', metavar='str',help='asfs#0, if omitted all will be procesed')
    parser.add_argument('-pd', '--pickledir', metavar='str',help='want to store a pickle of the data for debugging?')
    # add verboseprint function for extra info using verbose flag, ignore these 5 lines if you want
    
    args         = parser.parse_args()
    verbose      = True if args.verbose else False # use this to run segments of code via v/verbose flag
    v_print      = print if verbose else lambda *a, **k: None     # placeholder
    verboseprint = v_print # use this function to print a line if the -v/--verbose flag is provided
    
    global data_dir, in_dir, out_dir # make data available
    if args.path: data_dir = args.path
    else: data_dir = '/PSL/Observations/Campaigns/SPLASH/'
    
    if args.station: flux_stations = args.station.split(',')
    else: flux_stations = ['asfs30', 'asfs50']
 
    if args.pickledir: pickle_dir=args.pickledir
    else: pickle_dir=False
        
    def printline(startline='',endline=''):
        print('{}--------------------------------------------------------------------------------------------{}'
              .format(startline, endline))

    global start_time, end_time
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y%m%d')
    else:
        # make the data processing start yesterday! i.e. process only most recent full day of data
        start_time = epoch_time.today() # any datetime object can provide current time
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, day=start_time.day)
    if args.end_time:
        end_time = datetime.strptime(args.end_time, '%Y%m%d')
    else:
        end_time = epoch_time.today() # any datetime object can provide current time
        end_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, day=start_time.day)
        
    # expand the load by 1 day to facilite gps processing    
    start_time = start_time-timedelta(1)
    end_time = end_time+timedelta(1)

    print('The first day of data we will process data is:     %s' % str(start_time+timedelta(1)))
    print('The last day we will process data is:              %s\n\n' % str(end_time-timedelta(1)))
    printline()

    # thresholds! limits that can warn you about bad data!
    # these aren't used yet but should be used to warn about spurious data
    lat_thresh        = (0   ,90)        # limits area where stirt_targ_lo tion, they've been everywhere
    hdg_thresh_lo     = (0    ,360)      # limits on gps heading
    sr50d             = (0.6    ,3.5)    # distance limits from SR50 to surface [m]; install height -1 m or +0.5
    sr50_qc           = (170  ,300)      # reported "quality numbers" 0-151=can't read dist;
                                         # 210-300=reduced signal; 300+=high uncertainty
    irt_targ          = (-45  ,55)       # IRT surface brightness temperature limits [Celsius]
    flxp              = (-100 ,250)      # minimum and maximum conductive heat flux (W/m2)
    T_thresh          = (-35  ,30)       # minimum & maximum air temperatures (C)
    rh_thresh         = (5   ,110)       # relative humidity (#)
    p_thresh          = (675  ,1100)     # air pressure
    ws_thresh         = (0    ,40)       # wind speed from sonics (m/s)
    lic_co2sig_thresh = (94   ,105)      # rough estimate of minimum CO2 signal value corresponding to
                                         # optically-clean window. < 90 = needs cleaned (e.g., salt residue); < ~80?? = ice!
    lic_h2o           = (1e-3 ,50)       # Licor h2o [mg/m3]
    lic_co2           = (300  ,1000)     # Licor co2 [g/m3]
    max_bad_paths     = (0.01 ,1)        # METEK: maximum [fraction] of bad paths allowed. (0.01 = 1%), but is
                                         # actually in 1/9 increments. This just says we require all paths to be usable.
    incl_range        = (-90  ,90)       # The inclinometer on the metek
    met_t             = T_thresh         # Vaisala air temperature [C]
    met_rh            = rh_thresh        # Vaisala relative humidity [#]
    met_p             = p_thresh         # Vaisala air pressure [hPa or ~mb]
    alt_lim           = (-5.1,10.1)      # largest range of +/- 3sigma on the altitude data between the stations
    cd_lim            = (-2.3e-3,1.5e-2) # drag coefficinet sanity check. really it can't be < 0, but a small negative
                                         # threshold allows for empiracally defined (from EC) 3 sigma noise distributed about 0.
                                         # QCRAD thresholds & coefficents    
    sw_range          = (-10  ,1600)     # SWD & SWU max [Wm^2]
    lw_range          = (125  ,600)      # LWD & LWU max [Wm^2] 
    lwu_sigmax        = 30               # LWU max 1 min std 
    lwd_sigmax        = 10               # LWD max 1 min std 
    D1                = 1.15             # SWD CCL for Alert, which seems to work well here too (less data than climatology to check against!)
    D5                = 0.9              # SWU. Bumbed up a little from 0.88 used at Alert.
    D11               = 0.62             # LWD<->Tair low (bumped up from 0.58 at Alert)
    D12               = 23               # LWD<->Tair high (Alert)
    D13               = 12               # LWU<->Tair Low side. A lot stricter than Alert; I think the tighter LWU is because of the consitent
    D14               = 28               # LWU<->Tair High side. A lot stricter than Alert; 
    D15               = 300              # LWU <-> LWD test. Low side. Stricter than Alert, as with above.
    D16               = 20               # LWU <-> LWD test. High side. As at Alert.
    A0                = 1.2              # albedo limit

    # official "start" times, defined as EFOY-on on install day per Station Book
    station_initial_start_time = {}
    station_initial_start_time['asfs30'] = datetime(2021,9,28,18,29,0) 
    station_initial_start_time['asfs50'] = datetime(2021,10,12,20,44,0) 

    # ###################################################################################################
    # various calibration params
        
    init_asfs30 = pd.DataFrame()
    init_asfs50 = pd.DataFrame()
    
                                                                       # ASFS 30 ---------------------------
    init_asfs30['init_date']  = [
                                station_initial_start_time['asfs30'],  # Installed at Kettle Ponds Annex distance (281.6 cm) and 2-m Tvais (12.82 C) at 1829 UTC Sept 28, 2021
                                datetime(2023,10,1,0,0)                # end
                                ]
        
    init_asfs30['init_dist']  = [
                                sqrt((12.82+K_offset)/K_offset)*281.6, # Official 281.6 cm: Measured 306 cm with the tape, 279 cm with the SR50. Further adjustment of 2.6 cm determined for BAMS article. 
                                nan                                    # end
                                ]
        
    init_asfs30['init_depth'] = [
                                0,                                     # Easy! Snow-free!
                                nan                                    # end
                                ]    
        
    init_asfs30['init_lat']   = [
                                38 + 56.3686/60,                       # Kettle Ponds Annex
                                nan                                    # end
                                ]   
    
    init_asfs30['init_lon']   = [
                                -(106 + 58.1781/60),                      # Kettle Ponds Annex
                                nan                                    # end
                                ]   
    
    init_asfs30['init_hdg']   = [
                                177,                                   # Kettle Ponds Annex
                                nan                                    # end
                                ]  
    
    init_asfs30['init_alt']   = [
                                2856,                                  # Kettle Ponds Annex
                                nan                                    # end
                                ]  
   
    init_asfs30['init_pfb0']   = [
                                0.0803,                                # Kettle Ponds Annex
                                nan                                    # end
                                ]    
    
    init_asfs30['init_pfb1']   = [
                                -0.0489,                               # Kettle Ponds Annex
                                nan                                    # end
                                ] 
    
    init_asfs30['init_pfb2']   = [
                                -0.1717,                               # Kettle Ponds Annex
                                nan                                    # end
                                ]   

    init_asfs30['init_loc']   = [
                                'KPSA',                                # Kettle Ponds Annex
                                nan                                    # end
                                ]                     



                                                                       # ASFS 50 ---------------------------
    init_asfs50['init_date']  = [
                                station_initial_start_time['asfs50'],  # Installed at Avery Picnic (302 cm) and 2-m Tvais (-0.16 C) at 2043 UTC Oct 12, 2019
                                datetime(2023,10,1,0,0)                # end
                                ]
        
    init_asfs50['init_dist']  = [
                                sqrt((-0.16+K_offset)/K_offset)*302,   # Official 302 cm: Measurement thought to be 306 cm, but 23 cm refinement form the nominal 279 cm at KP for BAMS 
                                nan                                    # end
                                ]
                                
    init_asfs50['init_depth'] = [
                                0,                                     # Easy! Snow-free!
                                nan                                    # end
                                ]    
    
    init_asfs50['init_lat']   = [
                                38 + 58.3455/60,                       # Avery Picnic
                                nan                                    # end
                                ]   
    
    init_asfs50['init_lon']   = [
                                -(106 + 59.8113/60),                      # Avery Picnic
                                nan                                    # end
                                ]   
    
    init_asfs50['init_hdg']   = [
                                174.7,                                 # Avery Picnic
                                nan                                    # end
                                ]   
   
    init_asfs50['init_alt']   = [
                                2933.5,                                # Avery Picnic
                                nan                                    # end
                                ]  
    
    init_asfs50['init_pfb0']   = [
                                0.0598,                                # Avery Picnic
                                nan                                    # end
                                ]    
    
    init_asfs50['init_pfb1']   = [
                                0.0217,                                # Avery Picnic
                                nan                                    # end
                                ] 
    
    init_asfs50['init_pfb2']   = [
                                -0.168,                                # Avery Picnic
                                nan                                    # end
                                ]   
        
    init_asfs50['init_loc']   = [
                                'AYP',                                 # Avery Picnic
                                nan                                    # end
                                ]       
    
    # Set the index
    init_asfs30.set_index(init_asfs30['init_date'],inplace=True)
    init_asfs50.set_index(init_asfs50['init_date'],inplace=True)

    init_data = {}
    init_data['asfs30'] = init_asfs30
    init_data['asfs50'] = init_asfs50
    
    # Metadata for the radiometer tilt correction. We need a priori (i.e., when level prior to change in orientation) knowledge 
    # of the offsets between Metek inclinometer and SWD. Dates are begining of period and values should be persisted until the 
    # next date. nan signifies no correction should be performed.
    
    # For SPLASH, no corrections are expected, but we leave this as a placeholder in case that changes.                      
    asfs30_tilt_data = pd.DataFrame(np.array([ # date               IncX_offset IncY_offset  
                                            [datetime(2021,9,1,0,0),    nan,     nan], # Beg. Kettle Ponds Annex SPLASH
                                            [datetime(2023,10,1,0,0),    nan,     nan], # End.
                                            ]),columns=['date', 'incx_offset', 'incy_offset'])

    asfs30_tilt_data.set_index(asfs30_tilt_data['date'],inplace=True)
    
    asfs50_tilt_data = pd.DataFrame(np.array([ # date               IncX_offset IncY_offset  
                                            [datetime(2021,9,1,0,0),   nan,     nan], # Beg. Avery Picnic SPLASH
                                            [datetime(2023,10,1,0,0),    nan,     nan], # End.
                                            ]),columns=['date', 'incx_offset', 'incy_offset'])
    
    asfs50_tilt_data.set_index(asfs50_tilt_data['date'],inplace=True)
    
    tilt_data = {}
    tilt_data['asfs30'] = asfs30_tilt_data
    tilt_data['asfs50'] = asfs50_tilt_data
 
    # program logic starts here, the logic flow goes like this:
    # #########################################################
    # 
    #     1) read in all days of slow data from netcdf level 1 files
    #     2) loop over days requested
    #     3) for each day, pull in fast data from netcdf level 1 files
    #     5) for each day, apply strict QC and derive desired products in level2 definitions
    #     6) for each day, write level2 netcdf files
    #     7) if desired, using QCed observations, produce turbulent flux data product
    #     8) all done
    #
    # ... here we go then
    # ##########################################################################################

    print("Getting data from level1 netcdf files for stations: {}!!".format(flux_stations))
    print("   and doing it in threads, hopefully this doesn't burn your lap")
    printline()

    # getting the "slow" raw/level1 data is done here, heavy lifting is in "get_slow_data()" 
    # the slow dataset is small, so we load it all at once, the fast has to be done for each day
    # ##########################################################################################
    # dataframes are in dicts with station name keys, parallellizing data ingesting, saves time.
    slow_data   = {}
    slow_atts   = {}
    slow_vars   = {}

    slow_q_dict = {}
    slow_atts_from_definitions, slow_vars_from_definitions = define_level1_slow()

    print(f"Retreiving data from netcdf files... {data_dir}")
    for curr_station in flux_stations:
    
        in_dir = data_dir+'/'+curr_station+'/1_level_ingest_'+curr_station+'/'      # where does level 1 data live?
        df_station, code_version = get_splash_data(curr_station, start_time, end_time, 1,
                                                   data_dir, 'slow', verbose, nthreads, pickle_dir)
        slow_data[curr_station] = df_station

        printline()
        verboseprint("\n===================================================")
        verboseprint("Data and observations provided by {}:".format(curr_station))
        verboseprint('===================================================')
        if verbose: slow_data[curr_station].info(verbose=True) # must be contained; 
        verboseprint("\n\n")
        verboseprint(slow_data[curr_station])

    # mention big data gaps, good sanity check, using battvolt, because if battery voltage is missing...
    if verbose: 
        for curr_station in  flux_stations:
            curr_slow_data = slow_data[curr_station]

            try: 
                bv = curr_slow_data["batt_volt_Avg"]
            except:
                print(f"\n No data for {curr_station} for your requested range\n")
            threshold   = 60 # warn if the station was down for more than 60 minutes
            nan_groups  = bv.isnull().astype(int).groupby(bv.notnull().astype(int).cumsum()).cumsum()
            mins_down   = np.sum( nan_groups > 0 )

            prev_val   = 0 
            if np.sum(nan_groups > threshold) > 0:
                perc_down = round(((bv.size-mins_down)/bv.size)*100, 3)
                print(f"\nFor your time range, the {curr_station} was down for a total of {mins_down} minutes...")
                print(f"... which gives an uptime of approx {perc_down}% over the period data exists")

            else:
                print("\nStation {} was alive for the entire time range you requested!! Not bad... "
                      .format(curr_station, threshold))

    # first rename some columns to something we like better in way that deletes the old version
    # so we make sure not to perform operations on the wrong version below
                        # old name              : new name
    for curr_station in  flux_stations:
        slow_data[curr_station].rename(columns={'sr50_dist_Avg'         : 'sr50_dist'               , \
                                                'vaisala_P_Avg'         : 'atmos_pressure'          , \
                                                'vaisala_T_Avg'         : 'temp'                    , \
                                                'vaisala_Td_Avg'        : 'dew_point'               , \
                                                'vaisala_RH_Avg'        : 'rh'                      , \
                                                'apogee_body_T_Avg'     : 'body_T_IRT'              , \
                                                'apogee_targ_T_Avg'     : 'brightness_temp_surface' , \
                                                'fp_A_Wm2_Avg'          : 'subsurface_heat_flux_A'  , \
                                                'fp_B_Wm2_Avg'          : 'subsurface_heat_flux_B'  , \
                                                'licor_h2o_Avg'         : 'h2o_licor'               , \
                                                'licor_co2_Avg'         : 'co2_licor'               , \
                                                'licor_co2_str_out_Avg' : 'co2_signal_licor'        , \
                                                'sr30_swd_IrrC_Avg'     : 'down_short_hemisp'       , \
                                                'sr30_swu_IrrC_Avg'     : 'up_short_hemisp'         , \
                                                },inplace=True)
    
        ########################################  # # Process the GPS # #  #############################################
    # I'm going to do all the gps qc up front mostly because I need to smooth the heading before we split individual days  
    print('\n---------------------------------------------------------------------------------------------\n')            
    def process_gps(curr_station, sd):
        
        # For SPLASH, both sleds are stationary. No need for time series of GPS.
        # Values reported below are constants determined from the raw data in autumn 2021.
                
        # Get the current station's initialzation dataframe. ...eval...sorry, it had to be
        init_data[curr_station] = init_data[curr_station].reindex(method='pad',index=sd.index)
        print('Recording GPS metadata for '+curr_station) 
        
        sd['lon'] = init_data[curr_station]['init_lon']
        sd['lat'] = init_data[curr_station]['init_lat']
        sd['heading'] = init_data[curr_station]['init_hdg']
        sd['gps_alt_Avg'] = init_data[curr_station]['init_alt']
        
    
        return (sd, True)


    # IR20  radiometer calibration coefficients
    coef_S_down = {'asfs30': 12.15/1000 , 'asfs50' : 12.02/1000} 
    coef_a_down = {'asfs30': -16.36e-6  , 'asfs50' : -16.32e-6}  
    coef_b_down = {'asfs30': 2.62e-3    , 'asfs50' : 2.54e-3}    
    coef_c_down = {'asfs30': 0.9541     , 'asfs50' : 0.9557}     

    coef_S_up   = {'asfs30': 12.48/1000 , 'asfs50' : 12.01/1000} 
    coef_a_up   = {'asfs30': -16.25e-6  , 'asfs50' : -16.58e-6}  
    coef_b_up   = {'asfs30': 2.62e-3    , 'asfs50' : 2.49e-3}    
    coef_c_up   = {'asfs30': 0.9541     , 'asfs50' : 0.9568}
    # Planar fit coefs        
    global PFb0, PFb1, PFb2
    PFb0=  init_data[curr_station]['init_pfb0'][0]
    PFb1=  init_data[curr_station]['init_pfb1'][0]
    PFb2=  init_data[curr_station]['init_pfb2'][0]

    # actually call the gps functions and recalibrate LW sensors, minor adjustments to plates
    for curr_station in flux_stations:
        slow_data[curr_station], return_status = process_gps (curr_station, slow_data[curr_station])
        station_data = slow_data[curr_station]
   
        c_Sd =coef_S_down[curr_station]
        c_ad =coef_a_down[curr_station] 
        c_bd =coef_b_down[curr_station] 
        c_cd =coef_c_down[curr_station]                    
        c_Su =coef_S_up[curr_station] 
        c_au =coef_a_up[curr_station] 
        c_bu =coef_b_up[curr_station] 
        c_cu =coef_c_up[curr_station] 

        # Recalibrate IR20s. The logger code saved the raw and offered a preliminary cal that ignored the minor terms. We can do better.
        term = (c_ad*station_data['ir20_lwd_DegC_Avg']**2 + c_bd*station_data['ir20_lwd_DegC_Avg'] + c_cd)
        station_data['down_long_hemisp'] = station_data['ir20_lwd_mV_Avg'] / (c_Sd * term) + sb * (station_data['ir20_lwd_DegC_Avg']+273.15)**4
        term = (c_au*station_data['ir20_lwu_DegC_Avg']**2 + c_bu*station_data['ir20_lwu_DegC_Avg'] + c_cu)
        station_data['up_long_hemisp'] = station_data['ir20_lwu_mV_Avg'] / (c_Su * term) + sb * (station_data['ir20_lwu_DegC_Avg']+273.15)**4


        slow_data[curr_station] = station_data



    # ###############################################################################
    # OK we have all the slow data, now loop over each day and get fast data for that
    # day and do the QC/processing you want. all calculations are done in this loop
    # ... here's the meat and potatoes!!!

    day_series = pd.date_range(start_time+timedelta(1), end_time-timedelta(1)) # data was requested for these days
    day_delta  = pd.to_timedelta(86399999999,unit='us')                        # we want to go up to but not including 00:00

    printline(startline='\n')
    print("\n We have retreived all slow data, now processing each day...\n")

    # #########################################################################################################
    # here's where we actually call the data crunching function. for each station we process days sequentially
    # *then* move on to the next station, function for daily processing defined below
    def process_station_day(curr_station, today, tomorrow, slow_data_today, day_q=None):
        try:
            data_to_return = [] # this function processes the requested day then returns processed DFs appended to this
                                # list, e.g. data_to_return(data_name='slow', slow_df, None) or data_to_return(data_name='turb', turb_df, win_len)

            printline(endline="\n")
            print("Retreiving level1 fast data for {} on {}\n".format(curr_station,today))

            # get fast data from netcdf files for daily processing
            fast_data_today, fd_version = get_splash_data(curr_station, today-timedelta(1), today+timedelta(1),
                                                          1, data_dir, 'fast', verbose=False, nthreads=1)
            
            # get soil data from netcdf files for daily processing
            soil_data_today, so_version = get_splash_data(curr_station, today-timedelta(1), today+timedelta(1),
                                                          1, data_dir, 'soil', verbose=False, nthreads=1)

            # shorthand to save some space/make code more legible
            fdt = fast_data_today[today-timedelta(hours=1):tomorrow+timedelta(hours=1)]   
            sdt = slow_data_today[today-timedelta(hours=1):tomorrow+timedelta(hours=1)] 
            sot = soil_data_today[today-timedelta(hours=1):tomorrow+timedelta(hours=1)]

            idt = init_data[curr_station][today:tomorrow]
            if len(fdt.index)<=1: # data warnings and sanity checks
                if sdt.empty:
                    fail_msg = " !!! No data available for {} on {} !!!".format(curr_station, fl.dstr(today))
                    print(fail_msg)
                    try: day_q.put([('fail',fail_msg)]); return [('fail',fail_msg)]
                    except: return [('fail',fail_msg)]
                print("... no fast data available for {} on {}... ".format(curr_station, fl.dstr(today)))
            if len(sdt.index)<=1 or sdt['PTemp_Avg'].isnull().values.all():
                fail_msg = "... no slow data available for {} on {}... ".format(curr_station, fl.dstr(today))
                print(fail_msg)
                try: day_q.put([('fail',fail_msg)]); return [('fail',fail_msg)]
                except: return [('fail',fail_msg)]
            printline(startline="\n")

            print("\nQuality controlling data for {} on {}".format(curr_station, today))                  

            # First remove any data before official start of station in October (there may be some indoor or Fedorov hold test data)
            sdt[:].loc[:station_initial_start_time[curr_station]]=nan    

            # met sensor ppl
            sdt['atmos_pressure'].mask((sdt['atmos_pressure']<p_thresh[0]) | (sdt['atmos_pressure']>p_thresh[1]) , inplace=True) 
            sdt['temp']          .mask((          sdt['temp']<T_thresh[0]) | (sdt['temp']>T_thresh[1])           , inplace=True) 
            sdt['dew_point']     .mask((     sdt['dew_point']<T_thresh[0]) | (sdt['dew_point']>T_thresh[1])      , inplace=True) 
            sdt['rh']            .mask((           sdt['rh']<rh_thresh[0]) | (sdt['rh']>rh_thresh[1])            , inplace=True) 

            # Ephemeris, set it up
            utime_in = np.array(sdt.index.astype(np.int64)/10**9)                   # unix time. sec since 1/1/70
            lat_in   = sdt['lat']                                                   # latitude
            lon_in   = sdt['lon']                                                   # latitude
            elv_in   = np.zeros(sdt.index.size)+2                                   # elevation shall be 2 m; deets are negligible
            pr_in    = sdt['atmos_pressure'].fillna(sdt['atmos_pressure'].median()) # mb 
            t_in     = sdt['temp'].fillna(sdt['temp'].median())                     # degC

            # est of atm ref at sunrise/set following U. S. Naval Observatory's Vector Astrometry Software
            # @ wikipedia https://en.wikipedia.org/wiki/Atmospheric_refraction. This really just
            # sets a flag for when to apply the refraction adjustment. 
            atm_ref  = ( 1.02 * 1/np.tan(np.deg2rad(0+(10.3/(0+5.11)))) ) * pr_in/1010 * (283/(273.15+t_in))/60 
            delt_in  = spa.calculate_deltat(sdt.index.year, sdt.index.month) # seconds. delta_t between terrestrial and UT1 time

            # do the thing
            app_zenith, zenith, app_elevation, elevation, azimuth, eot = spa.solar_position(utime_in,lat_in,lon_in,
                                                                                            elv_in,pr_in,t_in,delt_in,atm_ref)

            # write it out
            sdt['zenith_true']     = zenith
            sdt['azimuth']         = azimuth 

            # in matlab, there were rare instabilities in the Reda and Andreas algorithm that resulted in spikes
            # (a few per year). no idea if this is a problem in the python version, but lets make sure
            sdt['zenith_true']     = fl.despike(sdt['zenith_true']     ,2,5,'no')
            sdt['azimuth']         = fl.despike(sdt['azimuth']         ,2,5,'no')

            # IR20 ventilation bias. The IRT was heated with 1.5 W. If the ventilator fan was off, lab & field
            # analysis suggests that the heat was improperly diffused causing a positive bias in the instrument
            # calculated at 1.42 Wm2 in the field and 1.28 Wm2 in the lab. We will use the latter here.
            sdt['up_long_hemisp'].loc[sdt['ir20_lwu_fan_Avg'] < 400]   = sdt['up_long_hemisp']-1.28
            sdt['down_long_hemisp'].loc[sdt['ir20_lwd_fan_Avg'] < 400] = sdt['down_long_hemisp']-1.28
            
            # IR20 variance threshold
            sdt['down_long_hemisp']              .mask( sdt['ir20_lwd_Wm2_Std'] > lwd_sigmax ,    inplace=True) # ppl
            sdt['up_long_hemisp']                .mask( sdt['ir20_lwu_Wm2_Std'] > lwu_sigmax ,    inplace=True) # ppl

            # IRT QC
            sdt['body_T_IRT']              .mask( (sdt['body_T_IRT']<irt_targ[0])    | (sdt['body_T_IRT']>irt_targ[1]) ,    inplace=True) # ppl
            sdt['brightness_temp_surface'] .mask( (sdt['brightness_temp_surface']<irt_targ[0]) | (sdt['brightness_temp_surface']>irt_targ[1]) , inplace=True) # ppl

            sdt['body_T_IRT']              .mask( (sdt['temp']<-1) & (abs(sdt['body_T_IRT'])==0) ,    inplace=True) # reports spurious 0s sometimes
            sdt['brightness_temp_surface'] .mask( (sdt['temp']<-1) & (abs(sdt['brightness_temp_surface'])==0) , inplace=True) # reports spurious 0s sometimes

            sdt['body_T_IRT']              = fl.despike(sdt['body_T_IRT'],2,60,'yes')              # replace spikes outside 2C
            sdt['brightness_temp_surface'] = fl.despike(sdt['brightness_temp_surface'],2,60,'yes') # over 60 sec with 60 s median

            # Flux plate QC
            sdt['subsurface_heat_flux_A'].mask( (sdt['subsurface_heat_flux_A']<flxp[0]) | (sdt['subsurface_heat_flux_A']>flxp[1]) , inplace=True) # ppl
            sdt['subsurface_heat_flux_B'].mask( (sdt['subsurface_heat_flux_B']<flxp[0]) | (sdt['subsurface_heat_flux_B']>flxp[1]) , inplace=True) # ppl
            if curr_station == 'asfs30': 
                sdt['subsurface_heat_flux_B'] = sdt['subsurface_heat_flux_B']-15.3909 # bias-correcting the single-ended measurement plate
                

            # SR50
            sdt['sr50_dist'].mask( (sdt['sr50_qc_Avg']<sr50_qc[0]) | (sdt['sr50_qc_Avg']>sr50_qc[1]) , inplace=True) # ppl
            sdt['sr50_dist'].mask( (sdt['sr50_dist']<sr50d[0])     | (sdt['sr50_dist']>sr50d[1]) ,     inplace=True) # ppl

            sdt['sr50_dist']  = fl.despike(sdt['sr50_dist'],0.05,90,"no")
            sdt['sr50_dist']  = fl.despike(sdt['sr50_dist'],0.02,5,"no") # screen but do not replace
            # if the qc is high, say 210-300 I think there is intermittent icing. this seems to work.
            if sdt['sr50_qc_Avg'].mean() > 230:
                sdt['sr50_dist']  = fl.despike(sdt['sr50_dist'],0.05,720,"no")


            # clean up missing met data that comes in as '0' instead of NaN... good stuff
            zeros_list = ['rh', 'atmos_pressure', 'sr50_dist']
            for param in zeros_list: # make the zeros nans
                sdt[param] = np.where(sdt[param]==0.0, nan, sdt[param])

            temps_list = ['temp', 'brightness_temp_surface', 'body_T_IRT']
            for param in temps_list: # identify when T==0 is actually missing data, this takes some logic
                potential_inds  = np.where(sdt[param]==0.0)
                if potential_inds[0].size==0: continue # if empty, do nothing, this is unnecessary
                for ind in potential_inds[0]:
                    #ind = ind.item() # convert to native python type from np.int64, so we can index
                    lo = ind
                    hi = ind+15
                    T_nearby = sdt[param][lo:hi]
                    if np.any(T_nearby < -5) or np.any(T_nearby > 5):    # temps cant go from 0 to +/-5C in 5 minutes
                        sdt[param].iloc[ind] = nan
                    elif (sdt[param].iloc[lo:hi] == 0).all(): # no way all values for a minute are *exactly* 0
                        sdt[param].iloc[lo:hi] = nan

            # Radiation
            sdt = fl.qcrad(sdt,sw_range,lw_range,D1,D5,D11,D12,D13,D14,D15,D16,A0)       
            # Tilt correction 
            # get the tilt data
            sdt['incx_offset'] = tilt_data[curr_station]['incx_offset'].reindex(index=sdt.index,method='pad').astype('float')
            sdt['incy_offset'] = tilt_data[curr_station]['incy_offset'].reindex(index=sdt.index,method='pad').astype('float')

            #diffuse_flux = -1 # we don't have an spn1 so we model the error. later we can use it if we have it

            # now run the correcting function      
            #if we_want_to_debug:
                #fl.pickle_function_args_for_debugging((sdt,diffuse_flux), pickle_dir, f'{today.strftime("%Y%m%d")}_tilt.pkl')
            #fl.tilt_corr(sdt,diffuse_flux) # modified sdt is returned

            # ###################################################################################################
            # derive some useful parameters that we want to write to the output file

            # compute RH wrt ice -- compute RHice(%) from RHw(%), Temperature(deg C), and pressure(mb)
            Td2, h2, a2, x2, Pw2, Pws2, rhi2 = fl.calc_humidity_ptu300(sdt['rh'],\
                                                                       sdt['temp']+K_offset,
                                                                       sdt['atmos_pressure'],
                                                                       0)
            sdt['rhi']                  = rhi2
            sdt['abs_humidity_vaisala'] = a2
            sdt['vapor_pressure']       = Pw2
            sdt['mixing_ratio']         = x2

            # snow depth in cm, corrected for temperature
            sdt['sr50_dist']  = sdt['sr50_dist']*sqrt((sdt['temp']+K_offset)/K_offset)
            sdt['snow_depth'] = idt['init_dist'] + (idt['init_depth']-sdt['sr50_dist']*100)

            # net radiation
            sdt['radiation_LWnet'] = sdt['down_long_hemisp']-sdt['up_long_hemisp']
            sdt['radiation_SWnet'] = sdt['down_short_hemisp']-sdt['up_short_hemisp']
            sdt['net_radiation']   = sdt['radiation_LWnet'] + sdt['radiation_SWnet'] 

            # make an Le time series that accounts for snow cover (Le of sublimation = vaporization+fusion) and snow free (Le of vaporization)
            # this will be binary, 0 = snow free, 1 = snow cover. Calculation comes later
            sdt['Le'] = 0 # all snow free
            if curr_station == 'asfs30':
                sdt['Le'].loc[datetime(2021,12,7,0,0,0):datetime(2022,5,4,0,0,0)]=1  
                sdt['Le'].loc[datetime(2022,10,23,0,0,0):datetime(2023,5,16,0,0,0)]=1 
            elif curr_station == 'asfs50':
                sdt['Le'].loc[datetime(2021,12,7,0,0,0):datetime(2022,5,12,0,0,0)]=1  
                sdt['Le'].loc[datetime(2022,10,23,0,0,0):datetime(2023,5,25,0,0,0)]=1 
            
            sdt['snow_flag']=sdt['Le'].copy(deep=True)
                
            # include a surface-aware emissivity    
            sdt['emis'] = emis_grass
            sdt['emis'].loc[sdt['Le']==1]=emis_snow

            #  ---- here we calculate the surface temperature. then we use it to do a small correction needed over melting snow (really! strong gradients are possible) then we recalc surface temp ----
            # surface skin temperature Persson et al. (2002) https://www.doi.org/10.1029/2000JC000705
            sdt['skin_temp_surface'] = (((sdt['up_long_hemisp']-(1-sdt['emis'])*sdt['down_long_hemisp'])/(sdt['emis']*sb))**0.25)-K_offset
            # make the correction
            dT = sdt['temp']-sdt['skin_temp_surface'] 
            #dT.loc[(dT < 0)] = 0 # if you only want to correct under stable states
            dT.fillna(0, inplace=True)
            sdt['up_long_hemisp'] = sdt['up_long_hemisp'] - 0.29193*dT-0.021642 # for 2 m
            
            # recalc
            sdt['skin_temp_surface'] = (((sdt['up_long_hemisp']-(1-sdt['emis'])*sdt['down_long_hemisp'])/(sdt['emis']*sb))**0.25)-K_offset            
            sdt['Le'].loc[sdt['skin_temp_surface'] > -0.2]=0 # if it is a melting snow, we assume the vaporization is from liquid not ice (i.e, evaporation)  
            
                                         
            # ###################################################################################################
            # all the 0.1 seconds today, for obs. we buffer by 1 hr for easy of po2 in turbulent fluxes below
            Hz10_today        = pd.date_range(today-pd.Timedelta(1,'hour'), tomorrow+pd.Timedelta(1,'hour'), freq='0.1S') 
            seconds_today     = pd.date_range(today, tomorrow, freq='S')    # all the seconds today, for obs
            minutes_today     = pd.date_range(today, tomorrow, freq='T')    # all the minutes today, for obs
            ten_minutes_today = pd.date_range(today, tomorrow, freq='10T')  # all the 10 minutes today, for obs

            #                              !! Important !!
            #   first resample to 10 Hz by averaging and reindexed to a continuous 10 Hz time grid (NaN at
            #   blackouts) of Lenth 60 min x 60 sec x 10 Hz = 36000 Later on (below) we will fill all missing
            #   times with the median of the (30 min?) flux sample.
            # ~~~~~~~~~~~~~~~~~~~~~ (2) Quality control ~~~~~~~~~~~~~~~~~~~~~~~~
            print("... quality controlling the fast data now")

            # check to see if fast data actually exists...
            # no data for param, sometimes fast is missing but slow isn't... very rare
            fast_var_list = ['metek_x', 'metek_y','metek_z','metek_T', 'metek_heatstatus',
                             'licor_h2o','licor_co2','licor_pr','licor_co2_str','licor_diag']
            for param in fast_var_list:
                try: test = fdt[param]
                except KeyError: fdt[param] = nan

            if fdt.empty: # create a fake dataframe
                nan_df = pd.DataFrame([[nan]*len(fdt.columns)], columns=fdt.columns)
                nan_df = nan_df.reindex(pd.DatetimeIndex([today]))
                fdt = nan_df.copy()             
                
            soil_var_list = ['VWC_5cm','Ka_5cm','T_5cm','BulkEC_5cm','VWC_10cm','Ka_10cm','T_10cm','BulkEC_10cm',
                             'VWC_20cm','Ka_20cm','T_20cm','BulkEC_20cm','VWC_30cm','Ka_30cm','T_30cm','BulkEC_30cm',
                             'VWC_40cm','Ka_40cm','T_40cm','BulkEC_40cm','VWC_50cm','Ka_50cm','T_50cm','BulkEC_50cm']
            for param in soil_var_list:
                try: test = sot[param]
                except KeyError: sot[param] = nan

            if sot.empty: # create a fake dataframe
                nan_df = pd.DataFrame([[nan]*len(sot.columns)], columns=sot.columns)
                nan_df = nan_df.reindex(pd.DatetimeIndex([today]))
                sot = nan_df.copy()
            
            sot.rename(columns={'VWC_5cm'        : 'soil_vwc_5cm'         , \
                                'Ka_5cm'         : 'soil_ka_5cm'          , \
                                'T_5cm'          : 'soil_t_5cm'           , \
                                'BulkEC_5cm'     : 'soil_ec_5cm'          , \
                                'VWC_10cm'       : 'soil_vwc_10cm'        , \
                                'Ka_10cm'        : 'soil_ka_10cm'         , \
                                'T_10cm'         : 'soil_t_10cm'          , \
                                'BulkEC_10cm'    : 'soil_ec_10cm'         , \
                                'VWC_20cm'       : 'soil_vwc_20cm'        , \
                                'Ka_20cm'        : 'soil_ka_20cm'         , \
                                'T_20cm'         : 'soil_t_20cm'          , \
                                'BulkEC_20cm'    : 'soil_ec_20cm'         , \
                                'VWC_30cm'       : 'soil_vwc_30cm'        , \
                                'Ka_30cm'        : 'soil_ka_30cm'         , \
                                'T_30cm'         : 'soil_t_30cm'          , \
                                'BulkEC_30cm'    : 'soil_ec_30cm'         , \
                                'VWC_40cm'       : 'soil_vwc_40cm'        , \
                                'Ka_40cm'        : 'soil_ka_40cm'         , \
                                'T_40cm'         : 'soil_t_40cm'          , \
                                'BulkEC_40cm'    : 'soil_ec_40cm'         , \
                                'VWC_50cm'       : 'soil_vwc_50cm'        , \
                                'Ka_50cm'        : 'soil_ka_50cm'         , \
                                'T_50cm'         : 'soil_t_50cm'          , \
                                'BulkEC_50cm'    : 'soil_ec_50cm'         , \
                                                    },inplace=True)
            
            # instead of sdt = pd.concat([sdt, sot], axis=1) which causes errors for cases that span dates with and without soil data, we are merging sot and sdt as below following the arm rad merge in the mosaic code
            
            soil_vars = ['soil_vwc_5cm','soil_ka_5cm','soil_t_5cm','soil_ec_5cm','soil_vwc_10cm','soil_ka_10cm','soil_t_10cm','soil_ec_10cm', \
                        'soil_vwc_20cm','soil_ka_20cm','soil_t_20cm','soil_ec_20cm','soil_vwc_30cm','soil_ka_30cm','soil_t_30cm','soil_ec_30cm', \
                        'soil_vwc_40cm','soil_ka_40cm','soil_t_40cm','soil_ec_40cm','soil_vwc_50cm','soil_ka_50cm','soil_t_50cm','soil_ec_50cm']
 
            sot_data = sot.sort_index(); 
            prev_len = len(sot_data)
            sot_data = sot_data.drop_duplicates(); 
            drop_len = len(sot_data)
            sot_inds = sot_data.index
            sdt = sdt.sort_index(); 
            sdt_inds = sdt.index

            not_present, index_map = compare_indexes(sot_inds, sdt_inds)
            sdt_map = np.array(index_map[0]) 
            sot_map  = np.array(index_map[1]) 
            
            # now we have to actually *put* the soil data into the slow_data dataframe at the mapped indices
            for iv, rv in enumerate(soil_vars):
                val_arr = np.array([nan]*len(sdt_inds))
                val_arr[sdt_map] = sot_data[rv].values[sot_map]
                sdt[rv] = val_arr
                sdt[rv] = sdt[rv].interpolate()
                
            # # # SOIL FLUX CALCULATION # # # 
            #
            # Calculate the conductive flux C, estimated at the surface 
            
            # First set some constants
            dt=60*10 # time delta for a 30 min in seconds
            if curr_station == 'asfs30':
                fq=0.35 # sand fraction at Kettle Ponds Annex (CSU SPUR results)
                fc=0.22 # clay fraction at Kettle Ponds Annex (CSU SPUR results)
                pb = 1.14 # dry soil density at Kettle Ponds Annex, g/cm3
                zz = 0.1 # depth of value
                kdry = 0.21 # dry soil thermal effective thermal conductivity at Kettle Ponds Annex measured this      % else 1/(6.757-7.089*log(pb));
                vwcfill = 0.11 # mean unfrozen 5/10 cm vwc to be a surrogate for missing vwc in calc of keff
                alpha = 0.67*fc+0.24
                beta1 = (np.log(0.7050-kdry)+0.05**-alpha) # Peng Eq 9 using samples. 
                beta2 =  1.97*fq+1.87*pb-1.36*fq*pb-0.95
                beta  = (beta1+beta2)/2
                Tskin = sdt['skin_temp_surface'].copy(deep=True)
                Tskin.loc[sdt['snow_flag']==1]=sdt['soil_t_5cm']
            elif curr_station == 'asfs50':
                fq=0.40 # sand fraction at Avery Picnic (CSU SPUR results)
                fc=0.25 # clay fraction at Avery Picnic (CSU SPUR results)
                pb = 1.03 # dry soil density at Avery Picnic, g/cm3  
                zz = 0.07 # depth of value
                kdry = 0.62 # dry soil thermal effective thermal conductivity at Avery Picnic measured this      % else 1/(6.757-7.089*log(pb));
                vwcfill = 0.06 # mean unfrozen 5/10 cm vwc to be a surrogate for missing vwc in calc of keff
                alpha = 0.67*fc+0.24
                beta1 = (np.log(0.7050-kdry)+0.04**-alpha) # Peng Eq 9 using samples. 
                beta2 =  1.97*fq+1.87*pb-1.36*fq*pb-0.95
                beta  = (beta1+beta2)/2
                Tskin = sdt['skin_temp_surface'].copy(deep=True)
                #T2 = sdt['brightness_temp_surface']         
                #downT = (sdt['down_long_hemisp']/sb)**0.25
                #Tskin = ((((T2+K_offset)**4-(1-sdt['emis'])*downT) / sdt['emis'])**0.25)-K_offset
                Tskin.loc[sdt['snow_flag']==1]=sdt['soil_t_5cm']
                          
            vwc = (sdt['soil_vwc_5cm']+sdt['soil_vwc_10cm'])/2  
            vwc.loc[vwc.isnull()] = vwcfill
            Cv = 1e3*(-0.224 -0.00561*((fq+fc)*100) +0.753*(pb*1000) +5.81*vwc*1000) # Eq 10: https://doi.org/10.1016/S1537-5110(03)00112-0  note that vwc*1000 = gwc*pb*1000 referring to Eq. 10
            keff = (kdry + np.exp(beta - vwc**-alpha))
            # calculate storage (S) and conductive flux (C). S is calculated in rolling 10 min intervals 
            sdt['s_soil']=-Cv*((((Tskin+K_offset)+(sdt['soil_t_5cm']+K_offset)+(sdt['soil_t_10cm']+K_offset))/3).diff(periods=10))/dt*-zz 
            sdt['c_soil'] = keff*((Tskin+K_offset)-(sdt['soil_t_10cm']+K_offset))/zz
            sdt['k_eff_soil']=keff
            
            # correct the flux plates for deflection error (Sauer 2007; Morgensen 1970)
            coef_alpha = 1.92
            coef_r = 5/((np.pi*40**2)**0.5)
            plate_k = 0.76
            gterm = 1 / (1 - coef_alpha*coef_r*(1-sdt['k_eff_soil']/plate_k))
            sdt['subsurface_heat_flux_A'] = sdt['subsurface_heat_flux_A'] / gterm
            sdt['subsurface_heat_flux_B'] = sdt['subsurface_heat_flux_B'] / gterm
                            
            metek_list = ['metek_x', 'metek_y','metek_z','metek_T']
            for param in metek_list: # identify when T==0 is actually missing data, this takes some logic
                potential_inds  = np.where(fdt[param]==0.0)

                if potential_inds[0].size==0: continue # if empty, do nothing, this is unnecessary
                if potential_inds[0].size>100000:
                    print("!!! there were a lot of zeros in your fast data, this shouldn't happen often !!!")
                    print("!!! {}% of {} was zero today!!!".format(round((potential_inds[0].size/1728000)*100,4),param))
                    split_val = 10000
                else: split_val=200

                while split_val>199: 
                    ind = 0
                    while ind < len(potential_inds[0]):
                        curr_ind = potential_inds[0][ind]
                        hi = int(curr_ind+(split_val))
                        if hi >= len(fdt[param]): hi=len(fdt[param])-1
                        vals_nearby = fdt[param][curr_ind:hi]
                        if (fdt[param].iloc[curr_ind:hi] == 0).all(): # no way all values are consecutively *exactly* 0
                            fdt[param].iloc[curr_ind:hi] = nan
                            ind = ind+split_val
                            continue
                        else:
                            ind = ind+1

                    potential_inds  = np.where(fdt[param]==0.0)
                    if split_val ==10000:
                        split_val = 200
                    else:
                        split_val = split_val -1
                #print("...ended with {} zeros in the array".format(len(potential_inds[0])))

            # I'm being a bit lazy here: no accouting for reasons data was rejected. For another day.
            # Chris said he was lazy first, now I'm being lazy by not making it up, sorry Chris

            # begin with bounding the Metek data to the physically-possible limits
            fdt ['metek_T']  [fdt['metek_T'] < T_thresh[0]] = nan
            fdt ['metek_T']  [fdt['metek_T'] > T_thresh[1]] = nan

            fdt ['metek_x']  [np.abs(fdt['metek_x'])  > ws_thresh[1]] = nan
            fdt ['metek_y']  [np.abs(fdt['metek_y'])  > ws_thresh[1]] = nan
            fdt ['metek_z']  [np.abs(fdt['metek_z'])  > ws_thresh[1]] = nan

            # Diagnostic: break up the diagnostic and search for bad paths. the diagnostic is as follows:
            # 1234567890123
            # 1000096313033
            #
            # char 1-2   = protocol stuff. 10 is actualy 010 and it says we are receiving instantaneous data from network. ignore
            # char 3-7   = data format. we use 96 = 00096
            # char 8     = heating operation mode. we set it to 3 = on but control internally for temp and data quality
                         # (ie dont operate the heater if you dont really have to)
            # char 9     = heating state, 0 = off, 1 = on, 2 = on but faulty
            # char 10    = number of unusable radial paths (max 9). we want this to be 0 and it is redundant with the next...
            # char 11-13 = percent of unusuable paths. in the example above, 3033 = 3 of 9 or 33% bad paths

            # We want to strip off the last 3 digits here and remove data that are not all 0s.  To do this
            # fast I will do it by subtracting off the top sig figs like below.  The minumum value is 1/9 so
            # I will set the threhsold a little > 0 for slop in precision. We could set this higher. Perhaps 1
            # or 2 bad paths is not so bad? Not sure.
            status = fdt['metek_heatstatus']
            bad_data = (status/1000-np.floor(status/1000)) >  max_bad_paths[0]
            fdt['metek_x'][bad_data]=nan
            fdt['metek_y'][bad_data]=nan
            fdt['metek_z'][bad_data]=nan
            fdt['metek_T'][bad_data]=nan

            # And now Licor ####################################################
            #
            # Physically-possible limits
            fdt['licor_h2o'] .mask( (fdt['licor_h2o']<lic_h2o[0]) | (fdt['licor_h2o']>lic_h2o[1]) , inplace=True) # ppl
            fdt['licor_co2'] .mask( (fdt['licor_co2']<lic_co2[0]) | (fdt['licor_co2']>lic_co2[1]) , inplace=True) # ppl
            fdt['licor_pr']  .mask( (fdt['licor_pr']<p_thresh[0]) | (fdt['licor_pr']>p_thresh[1]) , inplace=True) # ppl

            # CO2 signal strength is a measure of window cleanliness applicable to CO2 and H2O vars
            # first map the signal strength onto the fast data since it is empty in the fast files
            fdt['licor_co2_str'] = sdt['co2_signal_licor'].reindex(fdt.index).interpolate()
            fdt['licor_h2o'].mask( (fdt['licor_co2_str']<lic_co2sig_thresh[0]), inplace=True) # ppl
            fdt['licor_co2'].mask( (fdt['licor_co2_str']<lic_co2sig_thresh[0]), inplace=True) # ppl

            # The diagnostic is coded                                       
            print("... decoding Licor diagnostics.")

            pll, detector_temp, chopper_temp = fl.decode_licor_diag(fdt['licor_diag'])
            # Phase Lock Loop. Optical filter wheel rotating normally if 1, else "abnormal"
            bad_pll = pll == 0
            # If 0, detector temp has drifted too far from set point. Should yield a bad calibration, I think
            bad_dt = detector_temp == 0
            # Ditto for the chopper housing temp
            bad_ct = chopper_temp == 0
            # Get rid of diag QC failures
            fdt['licor_h2o'][bad_pll] = nan
            fdt['licor_co2'][bad_pll] = nan
            fdt['licor_h2o'][bad_dt]  = nan
            fdt['licor_co2'][bad_dt]  = nan
            fdt['licor_h2o'][bad_ct]  = nan
            fdt['licor_co2'][bad_ct]  = nan

            # Despike: meant to replace despik.m by Fairall. Works a little different tho
            #   Here screens +/-5 m/s outliers relative to a running 1 min median
            #
            #   args go like return = despike(input,oulier_threshold_in_m/s,window_length_in_n_samples)
            #
            #   !!!! Replaces failures with the median of the window !!!!
            #
            fdt['metek_x'] = fl.despike(fdt['metek_x'],5,1200,'yes')
            fdt['metek_y'] = fl.despike(fdt['metek_y'],5,1200,'yes')
            fdt['metek_z'] = fl.despike(fdt['metek_z'],5,1200,'yes')
            fdt['metek_T'] = fl.despike(fdt['metek_T'],5,1200,'yes')           
            fdt['licor_h2o'] = fl.despike(fdt['licor_h2o'],0.5,1200,'yes')
            fdt['licor_co2'] = fl.despike(fdt['licor_co2'],50,1200,'yes')

            # ~~~~~~~~~~~~~~~~~~~~~~~ (3) Resample  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print('... resampling 20 Hz -> 10 Hz.')
            #
            # 20 Hz irregular grid -> 10 Hz regular grid
            #
            # The method is to first resample the 20 Hz data to a 10 Hz regular
            # grid using the average of the (expect n=2) points at each 0.1s
            # interval. Then the result is indexed onto a complete grid for the
            # whole day, which is nominally 1 hour = 36000 samples at 10 Hz
            # Missing data (like NOAA Services blackouts) are nan

            fdt_10hz = fdt.resample('100ms').mean()

            fdt_10hz_ri = fdt_10hz.reindex(index=Hz10_today, method='nearest', tolerance='50ms')
            fdt_10hz = fdt_10hz_ri

            # ~~~~~~~~~~~~~~~~~ (4) Do the Tilt Rotation  ~~~~~~~~~~~~~~~~~~~~~~
            print("... cartesian tilt rotation. Translating body -> earth coordinates.")

            # This really only affects the slow interpretation of the data.
            # When we do the fluxes it will be a double rotation into the streamline that
            # implicitly accounts for deviations between body and earth
            #
            # The rotation is done in subroutine tilt_rotation, which is based on code from Chris Fairall et al.
            #
            # tilt_rotation(ct_phi,ct_theta,ct_psi,ct_up,ct_vp,ct_wp)
            #             ct_phi   = inclinometer roll angle (y)
            #             ct_theta = inclinometer pitchi angle (x)
            #             ct_psi   = yaw/heading/azimuth (z)
            #             ct_up    = y(u) wind
            #             ct_vp    = x(v) wind
            #             ct_zp    = z(w) wind
            #
            # Right-hand coordinate system convention:
            #             phi     =  inclinometer y is about the u axis
            #             theta   =  inclinometer x is about the v axis
            #             psi     =  azimuth        is about the z axis. the inclinometer does not measure this
            #                                                            despite what the manual may say (it's "aspirational").
            #             metek y -> earth u, +North
            #             metek x -> earth v, +West
            #             Have a look also at pg 21-23 of NEW_MANUAL_20190624_uSonic-3_Cage_MP_Manual for metek conventions.
            #             Pg 21 seems to have errors in the diagram?
            
            hdg = sdt['heading'].reindex(fdt_10hz.index).interpolate() # nominally, metek N is in line with the boom
          
            ct_u, ct_v, ct_w = fl.tilt_rotation(sdt['metek_InclY_Avg'].reindex(fdt_10hz.index).interpolate(),\
                                                sdt['metek_InclX_Avg'].reindex(fdt_10hz.index).interpolate(),\
                                                hdg,\
                                                fdt_10hz['metek_y'], fdt_10hz['metek_x'], fdt_10hz['metek_z'])

            # reassign corrected vals in meteorological convention, which involves swapping u and v and occurs in the following two blocks of 3 lines
            fdt_10hz['metek_x'] = ct_v 
            fdt_10hz['metek_y'] = ct_u
            fdt_10hz['metek_z'] = ct_w   

            # start referring to xyz as uvw now
            fdt_10hz.rename(columns={'metek_x':'metek_u'}, inplace=True)
            fdt_10hz.rename(columns={'metek_y':'metek_v'}, inplace=True)
            fdt_10hz.rename(columns={'metek_z':'metek_w'}, inplace=True)

            # !!
            # Now we recalculate the 1 min average wind direction and speed from the u and v velocities.
            # These values differ from the stats calcs (*_ws and *_wd) in two ways:
            #   (1) The underlying data has been quality controlled
            #   (2) We have rotated that sonic y,x,z into earth u,v,w
            #
            # I have modified the netCDF build to use *_ws_corr and *_wd_corr but have not removed the
            # original calculation because I think it is a nice opportunity for a sanity check. 
            print('... calculating a corrected set of slow wind speed and direction.')

            u_min = fdt_10hz['metek_u'].resample('1T',label='left').apply(fl.take_average)
            v_min = fdt_10hz['metek_v'].resample('1T',label='left').apply(fl.take_average)
            w_min = fdt_10hz['metek_w'].resample('1T',label='left').apply(fl.take_average)

            u_sigmin = fdt_10hz['metek_u'].resample('1T',label='left').std()
            v_sigmin = fdt_10hz['metek_v'].resample('1T',label='left').std()
            w_sigmin = fdt_10hz['metek_w'].resample('1T',label='left').std()
            
            ws = np.sqrt(u_min**2+v_min**2)
            wd = np.mod((np.arctan2(-u_min,-v_min)*180/np.pi),360)

            # ~~~~~~~~~~~~~~~~~~ (5) Recalculate Stats ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # !!  Sorry... This is a little messed up. The original stats are read from the NOAA Services stats
            # files, contents calculated from the raw data. But we have QC'd the data and changed the raw
            # values, so we need to update the stats. I do that here. But then was there ever any point in
            # reading the stats data in the first place?
            print('... recalculating NOAA Services style stats with corrected, rotated, and QCed values.')
            
            sdt['wspd_vec_mean']     = ws
            sdt['wdir_vec_mean']     = wd
            sdt['wspd_u_mean']       = u_min
            sdt['wspd_v_mean']       = v_min
            sdt['wspd_w_mean']       = w_min
            sdt['wspd_u_std']        = u_sigmin
            sdt['wspd_v_std']        = v_sigmin
            sdt['wspd_w_std']        = w_sigmin            
            sdt['temp_acoustic_std'] = fdt_10hz['metek_T'].resample('1T',label='left').std()
            sdt['temp_acoustic']     = fdt_10hz['metek_T'].resample('1T',label='left').mean()

            sdt['h2o_licor']         = fdt_10hz['licor_h2o'].resample('1T',label='left').mean()
            sdt['co2_licor']         = fdt_10hz['licor_co2'].resample('1T',label='left').mean()
            sdt['pr_licor']          = fdt_10hz['licor_pr'].resample('1T',label='left').mean()*10 # [to hPa]

            # ~~~~~~~~~~~~~~~~~~~~ (6) Flux Capacitor  ~~~~~~~~~~~~~~~~~~~~~~~~~
    
            if calc_fluxes == True and len(fdt.index) > 0:


                verboseprint('\nCalculating turbulent fluxes and associated MO parameters.')

                # Rotation to the streamline, FFT window segmentation, detrending,
                # hamming, and computation of power [welch] & cross spectral densities,
                # covariances and associated diagnostics and plots, as well as derived
                # variables (fluxes and stress parameters) are performed within a
                # sub-function called below.
                #
                # turbulence_data = fl.grachev_fluxcapacitor(sz, sonic_data, licor_data, h2o_units, co2_units, p, t, q, verbose=v)
                #       sz = instrument height
                #       sonic_data = dataframe of u,v,w winds
                #       licor_data = dataframe of h2o adn co2
                #       h2o_units = units of licor h2o, e.g., 'mmol/m3'
                #       co2_units = units of licor co2, e.g., 'mmol/m3'
                #       p = pressure in hPa, scaler
                #       t = air temperature in C, scaler
                #       q = vapor mixing ratio, scaler

                metek_10hz = fdt_10hz[['metek_u', 'metek_v', 'metek_w','metek_T']].copy()
                metek_10hz.rename(columns={\
                                           'metek_u':'u',
                                           'metek_v':'v',
                                           'metek_w':'w',
                                           'metek_T':'T'}, inplace=True)
                licor_10hz = fdt_10hz[['licor_h2o', 'licor_co2']].copy()

                # ######################################################################################
                # corrections to the high frequency component of the turbulence spectrum... the metek
                # sonics used seem to have correlated cross talk between T and w that results in biased
                # flux values with a dependency on frequency...
                #
                # this correction fixes that and is documented in the data paper, see comments in
                # functions_library
                metek_10hz = fl.fix_high_frequency(metek_10hz)

                turb_ec_data = {}

                # calculate before loop, used to modify height offsets below to be 'more correct'
                # snow depth calculation shouldn't/doesn't fail but catch the exception just in case
                try: 
                    snow_depth = sdt['snow_depth'][minutes_today].copy()  # get snow_depth, heights evolve in time
                    snow_depth[(np.abs(stats.zscore(snow_depth.values)) < 3)]   # remove weird outliers
                    snow_depth = snow_depth*0.01                                # convert to meters
                    snow_depth = snow_depth.rolling(30, min_periods=5).mean() # fill nans for bulk calc only
                    if snow_depth.isnull().all():
                        snow_depth[:]=0
                    snow_depth.loc[snow_depth.isnull()] = 0.
                    snow_depth.loc[snow_depth < 0] = 0.
                except Exception as ex: 
                    print(f"... calculating snow depth for {today} failed for some reason...")
                    snow_depth = pd.Series(0, index=sdt[minutes_today].index)

                for win_len in range(0,len(integ_time_turb_flux)):
                    integration_window = integ_time_turb_flux[win_len]
                    flux_freq_str = '{}T'.format(integration_window) # flux calc intervals
                    flux_time_today   = pd.date_range(today-timedelta(hours=1), tomorrow+timedelta(hours=1), freq=flux_freq_str) 
                    
                    # recalculate wind vectors to be saved with turbulence data  later
                    u_min  = metek_10hz['u'].resample(flux_freq_str, label='left').apply(fl.take_average)
                    v_min  = metek_10hz['v'].resample(flux_freq_str, label='left').apply(fl.take_average)
                    ws     = np.sqrt(u_min**2+v_min**2)
                    wd     = np.mod((np.arctan2(-u_min,-v_min)*180/np.pi),360)

                    turb_winds = pd.DataFrame()
                    turb_winds['wspd_vec_mean'] = ws
                    turb_winds['wdir_vec_mean'] = wd

                    ec_arg_list = []

                    first_success = True
                    nfailures = 0 
                    for time_i in range(0,len(flux_time_today)-1): # flux_time_today = a DatetimeIndex defined earlier and based
                                                                   # on integ_time_turb_flux, the integration window for the
                                                                   # calculations that is defined at the top of the code
                                        
                        if time_i % 24 == 0:
                            verboseprint(f'... turbulence integration across {flux_freq_str} for '+
                                         f'{flux_time_today[time_i].strftime("%m-%d-%Y %H")}h {curr_station}')

                        # Get the index, ind, of the metek frame that pertains to the present calculation A
                        # little tricky. We need to make sure we give it enough data to encompass the nearest
                        # power of 2: for 30 min fluxes this is ~27 min so you are good, but for 10 min fluxes
                        # it is 13.6 min so you need to give it more.

                        # We buffered the 10 Hz so that we can go outside the edge of "today" by up to an hour.
                        # It's a bit of a formality, but for general cleanliness we are going to
                        # center all fluxes to the nearest min so that e.g:
                        # 12:00-12:10 is actually 11:58 through 12:12
                        # 12:00-12:30 is actually 12:01 through 12:28     
                        po2_len  = np.ceil(2**round(np.log2(integration_window*60*10))/10/60) # @10Hz, min needed to cover nearet po2 [minutes]
                        t_win    = pd.Timedelta((po2_len-integration_window)/2,'minutes')
                        metek_in = metek_10hz.loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].copy()

                        # we need pressure and temperature and humidity
                        Pr_time_i    = sdt['atmos_pressure']    .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                        T_time_i     = sdt['temp']              .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                        Q_time_i     = sdt['mixing_ratio']      .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()/1000
                        Le_time_i    = sdt['Le']                .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mode()
                        snof_time_i  = sdt['snow_flag']          .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mode()
                        skin_time_i  = sdt['skin_temp_surface'] .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                        snow_depth_i = sdt['snow_depth']        .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                        if np.isnan(snow_depth_i): snow_depth_i = 0.0
                        if len(Le_time_i) == 0: Le_time_i = 0
                        if len(snof_time_i) == 0: snof_time_i = 0
                        
                        # get the licor data
                        licor_data = licor_10hz.loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].copy()

                        # make th1e turbulent flux calculations via Grachev module
                        v = False
                        if verbose: v = True;
                        sonic_z       = 4.62-snow_depth_i # what is sonic_z for the flux stations

                        fargs = (sonic_z, metek_in, licor_data, 'g/m3', 'mg/m3', Pr_time_i,
                                 T_time_i, Q_time_i, v, integration_window)
                        #if we_want_to_debug:
                            #fl.pickle_function_args_for_debugging(fargs, pickle_dir,
                            #                                      f'{today.strftime("%Y%m%d")}_{time_i}_grachev.pkl')
                        try:  
                            data = fl.grachev_fluxcapacitor(sonic_z, metek_in, licor_data, 'g/m3', 'mg/m3',Pr_time_i, T_time_i, Q_time_i, np.array(Le_time_i), PFb0, PFb1, PFb2, rotation_flag, v, integration_window)
                            # doubtless there is a better way to initialize this
                            if first_success:
                                turbulencetom = data; first_success = False
                            else: turbulencetom = turbulencetom.append(data)

                        except: 
                            nfailures+=1

                        # Sanity check on Cd. Ditch the run if it fails
                        #data[:].mask( (data['Cd'] < cd_lim[0])  | (data['Cd'] > cd_lim[1]) , inplace=True) 


                    if nfailures>5: print(f"!!! fluxcapacitor failed {nfailures} times for date {today}!!!")

                    try:
                        # now add the indexer datetime doohicky
                        turbulencetom.index = flux_time_today[0:-1] 
                        turb_cols = turbulencetom.keys()
                    except: 
                        #if we_want_to_debug:
                            #fl.pickle_function_args_for_debugging((sdt,metek_10hz,licor_10hz), pickle_dir, 
                            #                                      f'{today.strftime("%Y%m%d")}_fluxfail.pkl')
                        raise


                    # ugh. there are 2 dimensions to the spectral variables, but the spectra are smoothed. The smoothing routine
                    # is a bit strange in that is is dependent on the length of the window (to which it should be orthogonal!)
                    # and worse, is not obviously predictable...it groes in a for loop nested in a while loop that is seeded by
                    # a counter and limited by half the length of the window, but the growth is not entirely predictable and
                    # neither is the result so I can't preallocate the frequency vector. I need to talk to Andrey about this and
                    # I need a programatic solution to assigning a frequency dimension when pandas actually treats that
                    # dimension indpendently along the time dimension. I will search the data frame for instances of a frequency
                    # dim then assign times without it nan of that length. for days without a frequency dim I will assign it to
                    # be length of 2 arbitrarily so that the netcdf can be written. This is ugly.

                    # (1) figure out the length of the freq dim and how many times are missing. also, save the frequency itself
                    # or you will write that vector as nan later on...
                    missing_f_dim_ind = []
                    f_dim_len = 1 
                    for ii in range(0, np.array(turbulencetom['fs']).size):
                        len_var = np.array(turbulencetom['fs'][ii]).size
                        if len_var == 1:
                            missing_f_dim_ind.append(ii)
                        else:
                            f_dim_len = len_var
                            fs = turbulencetom['fs'][ii]
                            dfs = turbulencetom['dfs'][ii]

                    # (2) if missing times were found, fill with nans of the freq length you discovered. this happens on days
                    # when the instruents are turned on and also perhaps runs when missing data meant the flux_capacitor
                    # returned for lack of inputs
                    if f_dim_len > 0 and missing_f_dim_ind:        
                                                                    
                        # case we have no data we need to remake a nominal fs as a filler
                        if 'fs' not in locals(): 
                            fs = pd.DataFrame(np.zeros((60,1)),columns=['fs'])
                            fs = fs['fs']*nan
                        if 'dfs' not in locals(): 
                            dfs = pd.DataFrame(np.zeros((60,1)),columns=['dfs'])
                            dfs = dfs['dfs']*nan
                        

                        for ii in range(0,len(missing_f_dim_ind)):
                            # these are the array with multiple dims...  im filling the ones that are missing with nan (of fs in
                            # the case of fs...) such that they can form a proper and square array for the netcdf
                            turbulencetom['fs'][missing_f_dim_ind[ii]] = fs
                            turbulencetom['dfs'][missing_f_dim_ind[ii]] = dfs
                            turbulencetom['sUs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['sVs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['sWs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['sTs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['sqs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['scs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cWUs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cWVs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cWTs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cUTs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cVTs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cWqs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cUqs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cVqs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cWcs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cUcs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cVcs'][missing_f_dim_ind[ii]] = fs*nan
                            turbulencetom['cUVs'][missing_f_dim_ind[ii]] = fs*nan

                    turb_ec_data[win_len] = turbulencetom.copy()

                # calculate the bulk 
                print('... calculating bulk fluxes for day: {}'.format(today))

                # Input dataframe
                empty_data = np.zeros(np.size(sdt['mixing_ratio'][minutes_today]))
                bulk_input = pd.DataFrame()
                bulk_input['u']  = sdt['wspd_vec_mean'][minutes_today]     # wind speed                         (m/s)
                bulk_input['ts'] = sdt['skin_temp_surface'][minutes_today] # bulk water/ice surface tempetature (degC) 
                bulk_input['t']  = sdt['temp'][minutes_today]              # air temperature                    (degC) 
                bulk_input['Q']  = sdt['mixing_ratio'][minutes_today]/1000 # air moisture mixing ratio          (kg/kg)
                bulk_input['zi'] = empty_data+600                          # inversion height                   (m) wild guess
                bulk_input['P']  = sdt['atmos_pressure'][minutes_today]    # surface pressure                   (mb)
                bulk_input['zu'] = 4.62-snow_depth                         # height of anemometer               (m) 
                bulk_input['zu'].loc[bulk_input['zu'] < 0] = 0.01
                bulk_input['zt'] = 2.89-snow_depth                         # height of thermometer              (m)
                bulk_input['zt'].loc[bulk_input['zt'] < 0] = 0.01
                bulk_input['zq'] = 2.60-snow_depth                         # height of hygrometer               (m)   
                bulk_input['zq'].loc[bulk_input['zq'] < 0] = 0.01
                bulk_input['rh']  = sdt['rh'][minutes_today]/100           # relative humidity                  (unitless)
                bulk_input['vwc'] = sdt['soil_vwc_5cm'][minutes_today]     # volume water content               (m3/m3)
                bulk_input = bulk_input.resample(str(integration_window)+'min',label='left').apply(fl.take_average)

                # output dataframe
                empty_data = np.zeros(len(bulk_input))
                bulk = pd.DataFrame() 
                bulk['bulk_Hs']      = empty_data*nan # hsb: sensible heat flux (Wm-2)
                bulk['bulk_Hl']      = empty_data*nan # hlb: latent heat flux (Wm-2)
                bulk['bulk_tau']     = empty_data*nan # tau: stress                             (Pa)
                bulk['bulk_z0']      = empty_data*nan # zo: roughness length, veolicity              (m)
                bulk['bulk_z0t']     = empty_data*nan # zot:roughness length, temperature (m)
                bulk['bulk_z0q']     = empty_data*nan # zoq: roughness length, humidity (m)
                bulk['bulk_L']       = empty_data*nan # L: Obukhov length (m)       
                bulk['bulk_ustar']   = empty_data*nan # usr: friction velocity (sqrt(momentum flux)), ustar (m/s)
                bulk['bulk_tstar']   = empty_data*nan # tsr: temperature scale, tstar (K)
                bulk['bulk_qstar']   = empty_data*nan # qsr: specific humidity scale, qstar (kg/kg?)
                bulk['bulk_dter']    = empty_data*nan # dter
                bulk['bulk_dqer']    = empty_data*nan # dqer
                bulk['bulk_Hl_Webb'] = empty_data*nan # hl_webb: Webb density-corrected Hl (Wm-2)
                bulk['bulk_Cd']      = empty_data*nan # Cd: transfer coefficient for stress
                bulk['bulk_Ch']      = empty_data*nan # Ch: transfer coefficient for Hs
                bulk['bulk_Ce']      = empty_data*nan # Ce: transfer coefficient for Hl
                bulk['bulk_Cdn_10m'] = empty_data*nan # Cdn_10: 10 m neutral transfer coefficient for stress
                bulk['bulk_Chn_10m'] = empty_data*nan # Chn_10: 10 m neutral transfer coefficient for Hs
                bulk['bulk_Cen_10m'] = empty_data*nan # Cen_10: 10 m neutral transfer coefficient for Hl
                bulk['bulk_Rr']      = empty_data*nan # Reynolds number
                bulk['bulk_Rt']      = empty_data*nan # 
                bulk['bulk_Rq']      = empty_data*nan # 
                bulk=bulk.reindex(index=bulk_input.index)
                
                for ii in range(len(bulk)):
                    tmp = [bulk_input['u'][ii], bulk_input['ts'][ii], bulk_input['t'][ii], \
                           bulk_input['Q'][ii], bulk_input['zi'][ii], bulk_input['P'][ii], \
                           bulk_input['zu'][ii],bulk_input['zt'][ii], bulk_input['zq'][ii], \
                           bulk_input['rh'][ii],bulk_input['vwc'][ii]] 

                    if not any(np.isnan(tmp)):
                        bulkout = fl.cor_ice_A10(tmp,np.array(Le_time_i),np.array(snof_time_i),curr_station)
                        for hh in range(len(bulkout)):
                            if bulkout[13] < cd_lim[0] or bulkout[13] > cd_lim[1]:  # Sanity check on Cd. Ditch the whole run if it fails
                                bulk[bulk.columns[hh]][ii]=nan                      # for some reason this needs to be in a loop
                            else:
                                bulk[bulk.columns[hh]][ii]=bulkout[hh]

                for win_len in range(0,len(integ_time_turb_flux)):

                    # add this to the EC data, concat columns alongside each other without adding indexes
                    turbulencenew = pd.concat( [turb_ec_data[win_len], bulk, turb_winds], axis=1)  
                    data_to_return.append(('turb', turbulencenew.copy()[today:tomorrow], win_len))
                    if win_len < len(integ_time_turb_flux)-1: print('\n')
                    
            #out_dir   = '/PSL/Observations/Campaigns/SPLASH/'+curr_station+'/2_level_product/' # where will level 2 data written?
            out_dir = '/PSL/Observations/Campaigns/SPLASH/internal/30min/'+curr_station+'/2_level_product/'
            try: 
                trash_var = write_level2_10hz(curr_station, metek_10hz[today:tomorrow], licor_10hz[today:tomorrow], today, out_dir)
            except UnboundLocalError as ule:
                this_will_fail_if_no_fast_data = True

            data_to_return.append(('slow', sdt.copy()[today:tomorrow], None)) 

            try: day_q.put(data_to_return); return data_to_return
            except: return data_to_return
            

        except Exception as e: # this day failed with some sort of exception, but we want to keep processing 

            print(f"!!! processing of {curr_station} failed for {today}  !!!")
            print("==========================================================================================")
            print("Python traceback: \n\n")
            import traceback
            import sys
            print(traceback.format_exc())
            print("==========================================================================================")

            let_fail = True
            if let_fail: raise
            try: day_q.put([('trace', tbs)]); return [('trace', tbs)]
            except: return [('trace', tbs)]


    # used to store dataframes to be QCed/written after all days are processed,
    turb_data_dict = {}; slow_data_dict = {}
    for st in flux_stations: 
        turb_data_dict[st] = {}; slow_data_dict[st] = []
        for win_len in range(0,len(integ_time_turb_flux)):
            turb_data_dict[st][win_len] = []

    # call processing by day then station (allows threading for processing a single days data)
    failed_days = {}
    for curr_station in flux_stations:
        failed_days[curr_station] = []
        printline(endline=f"\n\n  Processing all requested days of data for {curr_station}\n\n"); printline()

        day_ind = -1*nthreads
        while day_ind < len(day_series): # loop over the days in the processing range and crunch away
            day_ind += nthreads

            q_list = []
            for day_i in range(day_ind, day_ind+nthreads): # with nthreads parallel processing
                if day_i >= len(day_series): continue

                today    = day_series[day_i]
                tomorrow = today+day_delta
                sd_today = slow_data[curr_station][today-timedelta(hours=1):tomorrow+timedelta(hours=1)]
                if len(sd_today[today:tomorrow]) == 0: continue # weird corner case where data begins tomorrow/ended yesterday

                q_today = Q()
                P(target=process_station_day, args=(curr_station, today, tomorrow, sd_today, q_today)).start()
                q_list.append((q_today, today))
                
            for qq, day in q_list: 
                try: df_tuple_list = qq.get(timeout=600)
                except:
                    import traceback
                    import sys
                    exc = traceback.format_exc()
                    failed_days[curr_station].append((day,exc))
                    df_tuple_list= None

                if type(df_tuple_list) != type([]): 
                    failed_days[curr_station].append((day,f"failed for undetermined reason, look at log {df_tuple_list}"))
                    continue

                for dft in df_tuple_list: 
                    return_status = dft[0]
                    if any(return_status in s for s in ['fail', 'trace']):
                        failed_days[curr_station].append((day, dft[1]))
                        break
                    elif dft[0] == 'slow':
                        slow_data_dict[curr_station].append(dft[1])
                    elif dft[0] == 'turb':
                        win_len = dft[2]
                        turb_data_dict[curr_station][win_len].append(dft[1])
                    else:
                        failed_days[curr_station].append((day,"failed for undetermined reason, look at log {dft}"))
                        break

    printline(endline=f"\n\n  Finished with data processing, now we QC and write out all files!!!"); printline()
    print("\n ... but first we have to concat the data and then QC, a bit slow")

    turb_all = {}; slow_all = {}; 
    for curr_station in flux_stations:
        slow_all[curr_station] = pd.concat( slow_data_dict[curr_station] )
        slow_all[curr_station] = slow_all[curr_station].sort_index() 
        turb_all[curr_station] = {}
        for win_len in range(0, len(integ_time_turb_flux)):
            turb_all[curr_station][win_len] = pd.concat( turb_data_dict[curr_station][win_len] ).sort_index()

        slow_all[curr_station] = qc_stations(slow_all[curr_station], curr_station)


    print(" ... done with concatting and QC, now we write! here's a sample of the output data:\n\n")
    for curr_station in flux_stations:
        print(slow_all[curr_station])
    
    
    def write_todays_data(curr_station, today, day_q):
        tomorrow = today+day_delta

        station_data = slow_all[curr_station][today:tomorrow].copy()
        
        out_dir = '/PSL/Observations/Campaigns/SPLASH/'+curr_station+'/2_level_product/' # where will level 2 data written?
        #out_dir   = data_dir+'/'+curr_station+'/2_level_product_'+curr_station+'/' # where will level 2 data written?
        
        wr, write_data = qc_asfs_winds(station_data.copy())

        # import pickle
        # met_args = [write_data.copy(), curr_station, today, "1min", out_dir]
        # pkl_file = open(f'./tests/{today.strftime("%Y%m%d")}_{curr_station}_met_data_write.pkl', 'wb')
        # pickle.dump(met_args, pkl_file)
        # pkl_file.close()

        trash_var = write_level2_netcdf(write_data.copy(), curr_station, today, "1min", out_dir)

        for win_len in range(0, len(integ_time_turb_flux)):
            integration_window = integ_time_turb_flux[win_len]
            fstr = f'{integ_time_turb_flux[win_len]}T' # pandas notation for timestep
            turb_data = turb_all[curr_station][win_len][today:tomorrow]

            # do averaging, a little weird, a little kludgy, a little annoying... whatever...
            # should have built the qc pipeline into the original code as a module and this would be cleaner

            data_list = []

            l2_atts, l2_cols = define_level2_variables(curr_station); qc_atts, qc_cols = define_qc_variables(include_turb=True)
            l2_cols = ['lat'] + ['lon'] + ['heading'] + ['gps_alt_Avg'] + l2_cols 
            l2_cols = l2_cols+qc_cols

            vector_vars = ['wspd_vec_mean', 'wdir_vec_mean']
            angle_vars  = ['heading', ] # others?

            for ivar, var_name in enumerate(l2_cols):
                try: 
                    if var_name.split('_')[-1] == 'qc':
                        data_list.append(fl.average_flags(station_data[var_name], fstr))
                    elif any(substr in var_name for substr in vector_vars):
                        data_list.append(turb_data[var_name]) # yank a few select variables out of turbulence, vestigial nonsense
                    elif any(substr in var_name for substr in angle_vars):
                        data_list.append(station_data[var_name].resample(fstr, label='left').apply(fl.take_average, is_angle=True))
                    elif 'soil_' in var_name:
                        data_list.append(station_data[var_name].resample(fstr, label='left').apply(fl.take_average, perc_allowed_missing=100))
                    else:
                        data_list.append(station_data[var_name].resample(fstr, label='left').apply(fl.take_average))
                except Exception as e: 
                    # this is a little silly, data didn't exist for var fill with nans so computation continues
                    print(f"... wait what/why/huh??? {var_name} — {e}")
                    data_list.append(pd.Series([np.nan]*len(station_data), index=station_data.index, name=var_name)\
                                     .resample(fstr, label='left').apply(fl.take_average))

            avged_data = pd.concat(data_list, axis=1)
            avged_data = avged_data[today:tomorrow]

            try:

                # import pickle
                # pkl_file = open(f'./tests/{today.strftime("%Y%m%d")}_{curr_station}_seb_qc_before.pkl', 'wb')
                # pickle.dump([avged_data,turb_data], pkl_file)
                # pkl_file.close()

                wr, avged_data = qc_asfs_winds(avged_data)

                avged_data, turb_data = qc_asfs_turb_data(avged_data.copy(), turb_data.copy())

                # for debugging the write function.... ugh
                # import pickle
                # seb_args = [avged_data.copy(), turb_data]
                # pkl_file = open(f'./tests/{today.strftime("%Y%m%d")}_{curr_station}_seb_qc_after.pkl', 'wb')
                # pickle.dump(seb_args, pkl_file)
                # pkl_file.close()
                trash_var = write_level2_netcdf(avged_data.copy(), curr_station, today,
                                                f"{integration_window}min", out_dir, turb_data)


            except: 
                print(f"!!! failed to qc and write turbulence data for {win_len} on {today} !!!")
                print("==========================================================================================")
                print("Python traceback: \n\n")
                import traceback
                import sys
                print(traceback.format_exc())
                print("==========================================================================================")
                #print(sys.exc_info()[2])


        day_q.put(True) 

    # call processing by day then station (allows threading for processing a single days data)
    for curr_station in flux_stations:
        printline(endline=f"\n\n  Writing all requested days of data for {curr_station}\n\n"); printline()
        q_list = []  # setup queue storage
        for i_day, today in enumerate(day_series): # loop over the days in the processing range and crunch away

            q_today = Q()
            P(target=write_todays_data, args=(curr_station, today, q_today)).start()
            q_list.append(q_today)

            if (i_day+1) % nthreads == 0 or today == day_series[-1]:
                for qq in q_list: qq.get()
                q_list = []

    printline()
    print("All done! Go check out your freshly baked files!!!")
    print(version_msg)
    printline()
    if any(len(fdays)>0 for fdays in failed_days.values()):
        print("\n\n ... but wait, there's more, the following days failed with exceptions: \n\n")
        printline()
        for curr_station in flux_stations: 
            for fday in failed_days[curr_station]:
                date, exception = fday
                print(f"... {date} for {curr_station} -- with:\n {exception}\n\n")


# do the stuff to write out the level1 files, set timestep equal to anything from "1min" to "XXmin"
# and we will average the native 1min data to that timestep. right now we are writing 1 and 10min files
def write_level2_netcdf(l2_data, curr_station, date, timestep, out_dir, turb_data=None):

    day_delta = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00
    tomorrow  = date+day_delta

    l2_atts, l2_cols = define_level2_variables(curr_station)

    all_missing     = True 
    first_loop      = True
    n_missing_denom = 0

    if l2_data.empty:
        print("... there was no data to write today {} for {} at station {}".format(date,timestep,curr_station))
        return False

    # get some useful missing data information for today and print it for the user
    for var_name, var_atts in l2_atts.items():
        try: dt = l2_data[var_name].dtype
        except KeyError as e: 
            print(" !!! no {} in data for {} ... does this make sense??".format(var_name, curr_station))
        perc_miss = fl.perc_missing(l2_data[var_name].values)
        if perc_miss < 100: all_missing = False
        if first_loop: 
            avg_missing = perc_miss
            first_loop=False
        elif perc_miss < 100: 
            avg_missing = avg_missing + perc_miss
            n_missing_denom += 1
    if n_missing_denom > 1: avg_missing = round(avg_missing/n_missing_denom,2)
    else: avg_missing = 100.

    print("... writing {} level2 for {} on {}, ~{}% of data is present".format(timestep, curr_station, date, 100-avg_missing))

    short_name = "met"
    if timestep != "1min":
        short_name = 'seb'
        
    file_str = 'sled{}.{}.{}.{}.{}.nc'.format(short_name,curr_station,lvlname,timestep,date.strftime('%Y%m%d.%H%M%S'))
    lev2_name  = '{}/{}'.format(out_dir, file_str)
 
    if short_name=="seb": 
        global_atts = define_global_atts(curr_station,"seb") # global atts for level 1 and level 2
    else:
        global_atts = define_global_atts(curr_station,"level2") # global atts for level 1 and level 2

    netcdf_lev2 = Dataset(lev2_name, 'w')# format='NETCDF4_CLASSIC')

    for att_name, att_val in global_atts.items(): # write global attributes 
        netcdf_lev2.setncattr(att_name, att_val)

    # put turbulence and 'slow' data together
    if isinstance(turb_data, type(pd.DataFrame())):
        turb_atts, turb_cols = define_turb_variables()
        qc_atts, qc_cols = define_qc_variables(include_turb=True)
    else:
        qc_atts, qc_cols = define_qc_variables()
        turb_atts = {}

    l2_atts.update(qc_atts) # combine qc atts/vars now
    l2_cols = l2_cols+qc_cols

    write_data_list = []
    fstr = '{}T'.format(timestep.rstrip("min"))
    
    if timestep != "1min":

        # we also need freq. dim for some turbulence vars and fix some object-oriented confusion
        # this loop effectively exists to normalize data in a way that the netcdf4 library 
        # can easily write them out, clean up weird dtypes==object, multi dimensional
        # missing data (time, freq) for turbulence calcultations, etc etc etc   
        first_exception = True
        for var_name, var_atts in turb_atts.items(): 
            try: turb_data[var_name]
            except KeyError as ke: 
                if var_name.split("_")[-1] == 'qc': continue; do_nothing = True # we don't fill in all qc variables yet
                else: raise 
                
            chosen_index = 0 # pick first time point to find the number of frequency datapoints written

            if turb_data[var_name].isna().all() or type(turb_data[var_name][chosen_index]) == type(float()):
                turb_data[var_name] = np.float64(turb_data[var_name])

            # this is weird and should only every happen if there was all nans in the fast data
            elif turb_data[var_name][chosen_index].dtype != np.dtype('float64') and turb_data[var_name][chosen_index].len > 1:

                if first_exception:
                    print(f"... something was strange about the fast data on {date} {ae}"); first_exception = False

                max_size = 0 
                for i in turb_data.index: 
                    try:
                        this_size = turb_data[var_name][i].size
                        if this_size > max_size:
                            max_size = this_size
                            chosen_index = i
                    except: do_nothing = True

                for i in turb_data.index: 
                    try: this_size = turb_data[var_name][i].size
                    except: 
                        turb_data[var_name][i] = pd.Series([nan]*max_size)

            # create variable, # dtype inferred from data file via pandas
            if 'fs' == var_name:
                netcdf_lev2.createDimension('freq', turb_data[var_name][chosen_index].size)   
            if 'dfs' == var_name:
                netcdf_lev2.createDimension('dfreq', turb_data[var_name][chosen_index].size)                   
 
    write_data = l2_data # vestigial, like many things

    # unlimited dimension to show that time is split over multiple files (makes dealing with data easier)
    netcdf_lev2.createDimension('time', None)

    dti = pd.DatetimeIndex(write_data.index.values)
    fstr = '{}T'.format(timestep.rstrip("min"))
    if timestep != "1min":
        dti = pd.date_range(date, tomorrow, freq=fstr)

    try:
        fms = write_data.index[0]
    except Exception as e:
        print("... something went really wrong with the indexing")
        print("... the code doesn't handle that currently")
        raise e

    # base_time, soil spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm =  np.datetime64(today_midnight)

    # first write the int base_time, the temporal distance from the UNIX epoch
    base = netcdf_lev2.createVariable('base_time', 'i') # seconds since
    base[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])      # seconds

    base_atts = {'string'     : '{}'.format(bot),
                 'long_name' : 'Base time since Epoch',
                 'units'     : 'seconds since {}'.format(et),
                 'ancillary_variables'  : 'time_offset',}
    for att_name, att_val in base_atts.items(): netcdf_lev2['base_time'].setncattr(att_name,att_val)

    # here we create the array and attributes for 'time'
    t_atts   = { 'units'     : 'seconds since {}'.format(tm),
                 'delta_t'   : '0000-00-00 00:01:00',
                 'long_name' : 'Time offset from midnight',
                 'calendar'  : 'standard',}


    bt_atts   = {'units'     : 'seconds since {}'.format(bot),
                     'delta_t'   : '0000-00-00 00:01:00',
                     'long_name' : 'Time offset from base_time',
                     'calendar'  : 'standard',}


    delta_ints = np.floor((dti - tm).total_seconds())      # seconds

    t_ind = pd.Int64Index(delta_ints)

    # set the time dimension and variable attributes to what's defined above
    t = netcdf_lev2.createVariable('time', 'd','time') # seconds since

    # now we create the array and attributes for 'time_offset'
    bt_delta_ints = np.floor((dti - bot).total_seconds())      # seconds

    bt_ind = pd.Int64Index(bt_delta_ints)

    # set the time dimension and variable attributes to what's defined above
    bt = netcdf_lev2.createVariable('time_offset', 'd','time') # seconds since

    # this try/except is vestigial, this bug should be fixed
    t[:]  = t_ind.values
    bt[:] = bt_ind.values

    for att_name, att_val in t_atts.items(): netcdf_lev2['time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts.items(): netcdf_lev2['time_offset'].setncattr(att_name,att_val)

    lats = netcdf_lev2.createVariable('lat', 'float64') # seconds since
    lats[:] = l2_data['lat'][0]
    lat_atts = {'long_name'     : 'latitude from gps at station',
                'cf_name'       : 'latitude',
                'instrument'    : 'Hemisphere V102',
                'methods'       : 'GPRMC, GPGGA, GPGZDA',
                'height'        : '2.8 m',
                'location'      : 'ASFS boom (top of station)'}
    for att_name, att_val in lat_atts.items(): netcdf_lev2['lat'].setncattr(att_name,att_val)
    
    lons = netcdf_lev2.createVariable('lon', 'float64') # seconds since
    lons[:] = l2_data['lon'][0]
    lon_atts = {'long_name'     : 'longitude from gps at station',
                'cf_name'       : 'longitude',
                'instrument'    : 'Hemisphere V102',
                'methods'       : 'GPRMC, GPGGA, GPGZDA',
                'height'        : '2.8 m',
                'location'      : 'ASFS boom (top of station)'}
    for att_name, att_val in lon_atts.items(): netcdf_lev2['lon'].setncattr(att_name,att_val)
    
    hdgs = netcdf_lev2.createVariable('heading', 'float64') # seconds since
    hdgs[:] = l2_data['heading'][0]
    hdg_atts = {'long_name'     : 'heading from gps at station',
                'cf_name'       : '',
                'instrument'    : 'Hemisphere V102',
                'methods'       : 'HEHDT',
                'height'        : '2.8 m',
                'location'      : 'ASFS boom (top of station)'}
    for att_name, att_val in hdg_atts.items(): netcdf_lev2['heading'].setncattr(att_name,att_val)

    alts = netcdf_lev2.createVariable('altitude', 'float64') # seconds since
    alts[:] = l2_data['gps_alt_Avg'][0]
    alt_atts = {'long_name'     : 'altitude from gps at station',
                'cf_name'       : '',
                'instrument'    : 'Hemisphere V102',
                'methods'       : 'GPGGA',
                'height'        : '2.8 m',
                'location'      : 'ASFS boom (top of station)'}
    for att_name, att_val in alt_atts.items(): netcdf_lev2['altitude'].setncattr(att_name,att_val)
    

    for var_name, var_atts in l2_atts.items():

        #try: var_dtype = write_data[var_name].dtype
        #except: 
        #    print(f"NOT WRITING VARIABLE {var_name} for {date} —— it didn't exist")
        #    continue
        if  var_name.split('_')[-1] == 'qc' or var_name == 'snow_flag':
            var_dtype = np.int32
        else: 
            var_dtype = np.float64

        perc_miss = fl.perc_missing(write_data[var_name])

        if  var_name.split('_')[-1] == 'qc' or var_name == 'snow_flag': #fl.column_is_ints(write_data[var_name]): #fl.column_is_ints(write_data[var_name]):
            var_dtype = np.int32
            fill_val  = def_fill_int
            var_tmp = write_data[var_name].values.astype(np.int32)
        else:
            fill_val  = def_fill_flt
            var_tmp = write_data[var_name].values.astype(np.int32)

        # all qc flags set to -1 for when corresponding variables are missing data
        exception_cols = {'bulk_qc': 'bulk_Hs', 'turbulence_qc': 'Hs', 'Hl_qc': 'Hl'}

        try:
            if var_name.split('_')[-1] == 'qc':
                fill_val = np.int32(-1)
                if not any(var_name in c for c in list(exception_cols.keys())): 
                    write_data.loc[write_data[var_name].isnull(), var_name] = 0
                    write_data.loc[write_data[var_name.rstrip('_qc')].isnull(), var_name] = fill_val
            
                else: 
                    write_data.loc[write_data[var_name].isnull(), var_name] = 0
                    write_data.loc[turb_data[exception_cols[var_name]].isnull(), var_name] = fill_val

        except Exception as e:
            print("Python traceback: \n\n")
            import traceback
            import sys
            print(traceback.format_exc())
            print("==========================================================================================\n")
            
            print(f"!!! failed to fill in qc var: {var_name}!!!\n !!! {e}")

        var  = netcdf_lev2.createVariable(var_name, var_dtype, 'time')

        # write atts to the var now
        for att_name, att_desc in var_atts.items(): netcdf_lev2[var_name].setncattr(att_name, att_desc)
        netcdf_lev2[var_name].setncattr('missing_value', fill_val)

        vtmp = write_data[var_name].copy()

        max_val = np.nanmax(vtmp.values) # masked array max/min/etc
        min_val = np.nanmin(vtmp.values)
        avg_val = np.nanmean(vtmp.values)
        
        if var_name.split('_')[-1] != 'qc':
            netcdf_lev2[var_name].setncattr('max_val', max_val)
            netcdf_lev2[var_name].setncattr('min_val', min_val)
            netcdf_lev2[var_name].setncattr('avg_val', avg_val)

        vtmp.fillna(fill_val, inplace=True)
        var[:] = vtmp.values

        # add a percent_missing attribute to give a first look at "data quality"
        netcdf_lev2[var_name].setncattr('percent_missing', perc_miss)

    # loop over all the data_out variables and save them to the netcdf along with their atts, etc
    ivar=0
    for var_name, var_atts in turb_atts.items():
        ivar+=1
        if not turb_data.empty: 
            # create variable, # dtype inferred from data file via pandas
            var_dtype = turb_data[var_name][0].dtype
            if turb_data[var_name][0].size == 1:

                # all qc flags set to -1 for when corresponding variables are missing data
                exception_cols = {'bulk_qc': 'bulk_Hs', 'turbulence_qc': 'Hs', 'Hl_qc': 'Hl'}

                var_turb  = netcdf_lev2.createVariable(var_name, var_dtype, 'time')

                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data)
                td = turb_data[var_name]
                td.fillna(def_fill_flt, inplace=True)
                var_turb[:] = td.values

            else:
                if 'fs' == var_name:  
                    var_turb  = netcdf_lev2.createVariable(var_name, var_dtype, ('freq'))
                    # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data). this is even stupider in multipple dimensions
                    td = turb_data[var_name][0]
                    try: 
                        td.fillna(def_fill_flt, inplace=True)
                        var_turb[:] = td.values
                    except: 
                        var_turb[:] = nan
                        print(f"... weird, {date} failed to get frequency data...")
                    
                elif 'dfs' == var_name:  
                    var_turb  = netcdf_lev2.createVariable(var_name, var_dtype, ('dfreq'))
                    # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data). this is even stupider in multipple dimensions
                    td = turb_data[var_name][0]
                    try: 
                        td.fillna(def_fill_flt, inplace=True)
                        var_turb[:] = td.values
                    except: 
                        var_turb[:] = nan
                        print(f"... weird, {date} failed to get frequency data...")

                else:   
                    var_turb  = netcdf_lev2.createVariable(var_name, var_dtype, ('time','freq'))

                    tmp_df = pd.DataFrame(index=turb_data.index)

                    try: 
                        # replaced some sketchy code loops with functional OO calls and list comprehensions
                        # put the timeseries into a temporary DF that's a simple timeseries, not an array of 'freq'
                        tmp_list    = [col.values for col in turb_data[var_name].values]
                        tmp_df      = pd.DataFrame(tmp_list, index=turb_data.index).fillna(def_fill_flt)
                        tmp         = tmp_df.to_numpy()
                        var_turb[:] = tmp         
                    except:
                        var_turb[:] = nan

        else:
            var_turb  = netcdf_lev2.createVariable(var_name, np.double(), 'time')
            var_turb[:] = nan

        # add variable descriptions from above to each file
        for att_name, att_desc in var_atts.items(): netcdf_lev2[var_name] .setncattr(att_name, att_desc)

        # add a percent_missing attribute to give a first look at "data quality"
        perc_miss = fl.perc_missing(var_turb)
        netcdf_lev2[var_name].setncattr('percent_missing', perc_miss)
        netcdf_lev2[var_name].setncattr('missing_value', def_fill_flt)

    netcdf_lev2.close() # close and write files for today

    return  True

def write_level2_10hz(curr_station, sonic_data, licor_data, date, out_dir):

    day_delta = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00
    tomorrow  = date+day_delta

    # these keys are the names of the groups in the netcdf files and the
    # strings in the tuple on the right will be the strings that we search for
    # in fast_atts to know which atts to apply to which group/variable, the
    # data is the value of the search string, a dict of dicts...
    sonic_data = sonic_data.add_prefix('metek_')

    inst_dict = {}
    inst_dict['metek'] = ('metek' , sonic_data)
    inst_dict['licor'] = ('licor' , licor_data)

    fast_atts, fast_vars = define_10hz_variables()

    print("... writing level1 fast data {}".format(date))

    # if there's no data, bale (had to look up how to spell bale)
    num_empty=0; 
    for inst in inst_dict:
        if inst_dict[inst][1].empty: num_empty+=1

    if num_empty == len(inst_dict): 
        print("!!! no data on day {}, returning from fast write without writing".format(date))
        return False

    file_str_fast = 'sledwind10hz.{}.{}.{}.nc'.format(curr_station,lvlname,date.strftime('%Y%m%d.%H%M%S'))
    
    lev2_10hz_name  = '{}/{}'.format(out_dir, file_str_fast)

    global_atts_fast = define_global_atts(curr_station, "10hz") # global atts for level 1 and level 2

    netcdf_10hz  = Dataset(lev2_10hz_name, 'w',zlib=True)

    for att_name, att_val in global_atts_fast.items(): # write the global attributes to fast
        netcdf_10hz.setncattr(att_name, att_val)

    # setup the appropriate time dimensions
    netcdf_10hz.createDimension(f'time', None)

    fms = sonic_data.index[0]
    beginning_of_time = fms 

    # base_time, soil spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et  = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm  =  np.datetime64(today_midnight)

    # first write the int base_time, the temporal distance from the UNIX epoch
    base_fast = netcdf_10hz.createVariable(f'base_time', 'i') # seconds since
    base_fast[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])

    base_atts = {'string'              : '{}'.format(bot),
                 'long_name'           : 'Base time since Epoch',
                 'units'               : 'seconds since {}'.format(et),
                 'ancillary_variables' : 'time_offset',}

    t_atts_fast   = {'units'     : 'milliseconds since {}'.format(tm),
                     'delta_t'   : '0000-00-00 00:00:00.001',
                     'long_name' : 'Time offset from midnight',
                     'calendar'  : 'standard',}

    bt_atts_fast   = {'units'    : 'milliseconds since {}'.format(bot),
                      'delta_t'   : '0000-00-00 00:00:00.001',
                      'long_name' : 'Time offset from base_time',
                      'calendar'  : 'standard',}

    bt_fast_dti = pd.DatetimeIndex(sonic_data.index.values)   
    fast_dti    = pd.DatetimeIndex(sonic_data.index.values)

    # set the time dimension and variable attributes to what's defined above
    t_fast      = netcdf_10hz.createVariable(f'time', 'd',f'time') 
    bt_fast     = netcdf_10hz.createVariable(f'time_offset', 'd',f'time') 

    bt_fast_delta_ints = np.floor((bt_fast_dti - bot).total_seconds()*1000)      # milliseconds
    fast_delta_ints    = np.floor((fast_dti - tm).total_seconds()*1000)      # milliseconds

    bt_fast_ind = pd.Int64Index(bt_fast_delta_ints)
    t_fast_ind  = pd.Int64Index(fast_delta_ints)

    t_fast[:]   = t_fast_ind.values
    bt_fast[:]  = bt_fast_ind.values

    for att_name, att_val in t_atts_fast.items():
        netcdf_10hz[f'time'].setncattr(att_name,att_val)
    for att_name, att_val in base_atts.items():
        netcdf_10hz[f'base_time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts_fast.items():
        netcdf_10hz[f'time_offset'].setncattr(att_name,att_val)

    # loop through the 4 instruments and set vars in each group based on the strings contained in those vars.
    # a bit sketchy but better than hardcoding things... probably. can at least be done in loop
    # and provides some flexibility for var name changes and such.
    for inst in inst_dict: # for instrument in instrument list create a group and stuff it with data/vars

        inst_str  = inst_dict[inst][0]
        inst_data = inst_dict[inst][1][date:tomorrow]

        #curr_group = netcdf_10hz.createGroup(inst) 

           
        used_vars = [] # used to sanity check and make sure no vars get left out of the file by using string ids
        for var_name, var_atts in fast_atts.items():
            if inst_str not in var_name: continue # check if this variable belongs to this instrument/group
            # if 'TIMESTAMP' in var_name:
            #     used_vars.append(var_name)
            #     continue # already did time stuff
            # actually, commenting out because maybe we should preserve the original timestamp integers???

            try: 
                var_dtype = inst_data[var_name].dtype
            except: 
                print(f" !!! variable {var_name} didn't exist in 10hz data, not writing any 10hz !!!")
                return False

            if var_name.split('_')[-1] == 'qc' or var_name == 'snow_flag': #fl.column_is_ints(inst_data[var_name]):
                var_dtype = np.int32
                fill_val  = def_fill_int
                inst_data[var_name].fillna(fill_val, inplace=True)
                var_tmp   = inst_data[var_name].values.astype(np.int32)

            else:
                fill_val  = def_fill_flt
                inst_data[var_name].fillna(fill_val, inplace=True)
                var_tmp   = inst_data[var_name].values
        
            try:
                var_fast = netcdf_10hz.createVariable(var_name, var_dtype, f'time', zlib=True)
                var_fast[:] = var_tmp # compress fast data via zlib=True

            except Exception as e:
                print("!!! something wrong with variable {} on date {} !!!".format(var_name, date))
                print(inst_data[var_name])
                print("!!! {} !!!".format(e))
                continue

            for att_name, att_desc in var_atts.items(): # write atts to the var now
                netcdf_10hz[var_name].setncattr(att_name, att_desc)

            # add a percent_missing attribute to give a first look at "data quality"
            perc_miss = fl.perc_missing(var_fast)
            netcdf_10hz[var_name].setncattr('percent_missing', perc_miss)
            netcdf_10hz[var_name].setncattr('missing_value'  , fill_val)
            used_vars.append(var_name)  # all done, move on to next variable

    print(f"... finished writing 10hz data for {date}")
    netcdf_10hz.close()
    return True

# do the stuff to write out the level1 files, set timestep equal to anything from "1min" to "XXmin"
# and we will average the native 1min data to that timestep. right now we are writing 1 and 10min files
def write_turb_netcdf(turb_data, curr_station, date, integration_window, win_len, out_dir):
    
    timestep = str(integration_window)+"min"

    day_delta = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00
    tomorrow  = date+day_delta

    turb_atts, turb_cols = define_turb_variables()

    if turb_data.empty:
        print("... there was no turbulence data to write today {} at station {}".format(date,curr_station))
        return False

    # get some useful missing data information for today and print it for the user
    if not turb_data.empty: avg_missing = (1-turb_data.iloc[:,0].notnull().count()/len(turb_data.iloc[:,1]))*100.
    #fl.perc_missing(turb_data.iloc[:,0].values)
    else: avg_missing = 100.

    print("... writing turbulence data for {} on {}, ~{}% of data is present".format(curr_station, date, 100-avg_missing))
    
    file_str = 'sled{}turb.level2.{}.{}.nc'.format(curr_station,timestep,date.strftime('%Y%m%d.%H%M%S'))

    turb_name  = '{}/{}'.format(out_dir, file_str)

    global_atts = define_global_atts(curr_station, "turb") # global atts for level 1 and level 2
    netcdf_turb = Dataset(turb_name, 'w', zlib=True)
    # output netcdf4_classic files, for backwards compatibility... can be changed later but has some useful
    # features when using the data with 'vintage' processing code. it's the netcdf3 api, wrapped in hdf5

    # !! sorry, i have a different set of globals for this file so it isnt in the file list
    for att_name, att_val in global_atts.items(): netcdf_turb.setncattr(att_name, att_val) 
    n_turb_in_day = np.int(24*60/integration_window)

    netcdf_turb.createDimension('time', None)

    try:
        fms = turb_data.index[0]
    except Exception as e:
        print("... something went really wrong with the indexing")
        print("... the code doesn't handle that currently")
        raise e

    # base_time, soil spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm =  np.datetime64(today_midnight)

    # first write the int base_time, the temporal distance from the UNIX epoch
    base = netcdf_turb.createVariable('base_time', 'i') # seconds since
    base[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])      # seconds

    base_atts = {'string'     : '{}'.format(bot),
                 'long_name' : 'Base time since Epoch',
                 'units'     : 'seconds since {}'.format(et),
                 'ancillary_variables'  : 'time_offset',}
    for att_name, att_val in base_atts.items(): netcdf_turb['base_time'].setncattr(att_name,att_val)

    if integ_time_turb_flux[win_len] < 10:
        delt_str = f"0000-00-00 00:0{integ_time_turb_flux[win_len]}:00"
    else:
        delt_str = f"0000-00-00 00:{integ_time_turb_flux[win_len]}:00"

    # here we create the array and attributes for 'time'
    t_atts   = {'units'     : 'seconds since {}'.format(tm),
                'delta_t'   : delt_str,
                'long_name' : 'Time offset from midnight',
                'calendar'  : 'standard',}


    bt_atts   = {'units'     : 'seconds since {}'.format(bot),
                 'delta_t'   : delt_str,
                 'long_name' : 'Time offset from base_time',
                 'calendar'  : 'standard',}

    dti = pd.DatetimeIndex(turb_data.index.values)
    delta_ints = np.floor((dti - tm).total_seconds())      # seconds

    t_ind = pd.Int64Index(delta_ints)

    # set the time dimension and variable attributes to what's defined above
    t = netcdf_turb.createVariable('time', 'd','time') # seconds since

    # now we create the array and attributes for 'time_offset'
    bt_dti = pd.DatetimeIndex(turb_data.index.values)   

    bt_delta_ints = np.floor((bt_dti - bot).total_seconds())      # seconds

    bt_ind = pd.Int64Index(bt_delta_ints)

    # set the time dimension and variable attributes to what's defined above
    bt = netcdf_turb.createVariable('time_offset', 'd','time') # seconds since

    # this try/except is vestigial, this bug should be fixed
    t[:]  = t_ind.values
    bt[:] = bt_ind.values

    for att_name, att_val in t_atts.items(): netcdf_turb['time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts.items(): netcdf_turb['time_offset'].setncattr(att_name,att_val)

    # loop over all the data_out variables and save them to the netcdf along with their atts, etc
    for var_name, var_atts in turb_atts.items():
         
        # seriously python, seriously????
        if turb_data[var_name].isnull().all():
            if turb_data[var_name].dtype == object: # happens when all fast data is missing...
                turb_data[var_name] = np.float64(turb_data[var_name])     
        elif turb_data[var_name][0].size > 1:
            if turb_data[var_name][0].dtype == object: # happens when all fast data is missing...
                turb_data[var_name] = np.float64(turb_data[var_name])
        else:         
            if turb_data[var_name].dtype == object: # happens when all fast data is missing...
                turb_data[var_name] = np.float64(turb_data[var_name]) 

        # create variable, # dtype inferred from data file via pandas
        var_dtype = turb_data[var_name][0].dtype
        if turb_data[var_name][0].size == 1:
            var_turb  = netcdf_turb.createVariable(var_name, var_dtype, 'time')
            turb_data[var_name].fillna(def_fill_flt, inplace=True)
            # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data)
            var_turb[:] = turb_data[var_name].values
        else:
            if 'fs' == var_name:  
                netcdf_turb.createDimension('freq', turb_data[var_name][0].size)   
                var_turb  = netcdf_turb.createVariable(var_name, var_dtype, ('freq'))
                turb_data[var_name][0].fillna(def_fill_flt, inplace=True)
                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data). this is even stupider in multipple dimensions
                var_turb[:] = turb_data[var_name][0].values    
            elif 'dfs' == var_name:  
                netcdf_turb.createDimension('dfreq', turb_data[var_name][0].size)   
                var_turb  = netcdf_turb.createVariable(var_name, var_dtype, ('dfreq'))
                turb_data[var_name][0].fillna(def_fill_flt, inplace=True)
                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data). this is even stupider in multipple dimensions
                var_turb[:] = turb_data[var_name][0].values      
            else:   
                var_turb  = netcdf_turb.createVariable(var_name, var_dtype, ('time','freq'))
                for jj in range(0,turb_data[var_name].size):
                    turb_data[var_name][jj].fillna(def_fill_flt, inplace=True)
                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data). this is even stupider in multipple dimensions
                tmp = np.empty([turb_data[var_name].size,turb_data[var_name][0].size])
                for jj in range(0,turb_data[var_name].size):
                    tmp[jj,:]=np.array(turb_data[var_name][jj])
                var_turb[:] = tmp         

        # add variable descriptions from above to each file
        for att_name, att_desc in var_atts.items(): netcdf_turb[var_name] .setncattr(att_name, att_desc)

        # add a percent_missing attribute to give a first loop at "data quality"
        perc_miss = fl.perc_missing(var_turb)
        max_val = np.nanmax(var_turb) # masked array max/min/etc
        min_val = np.nanmin(var_turb)
        avg_val = np.nanmean(var_turb)

        netcdf_turb[var_name].setncattr('max_val', max_val)
        netcdf_turb[var_name].setncattr('min_val', min_val)
        netcdf_turb[var_name].setncattr('avg_val', avg_val)
        netcdf_turb[var_name].setncattr('percent_missing', perc_miss)
        netcdf_turb[var_name].setncattr('missing_value', def_fill_flt)

    netcdf_turb.close() # close and write files for today
    return True
    
def compare_indexes(inds_sparse, inds_lush, guess_jump = 50):
    inds_not_present = [] 
    map_between = ([],[]) # tuple of matching indexes

    sl = len(inds_sparse)

    # we assume monotonicity
    if not inds_sparse.is_monotonic_increasing or not inds_lush.is_monotonic_increasing:
        raise Exception("indexes not sorted")

    sparse_i = 0; old_perc = 0 


    print("")
    n_missed_but_not_really = 0 
    range_inds = iter(range(len(inds_sparse)))
    #for lush_i in range_inds:
    for sparse_i in range_inds:
        
        if sparse_i % 1000 == 0: print(f"... aligning {round((sparse_i/sl)*100,4)}% of soil/flux data", end='\r')

        sparse_date = inds_sparse[sparse_i] # get the actual timestamp to compare
        try: 
            lush_i = inds_lush.get_loc(sparse_date)
            map_between[0].append(lush_i)
            map_between[1].append(sparse_i)
        except Exception:
            inds_not_present.append(sparse_date)

    print(f"... aligned 100.000% of soil/flux \n", end='\r')

    #print(f"!!!!!! there were these indexes weirdly skipped {n_missed_but_not_really} !!!!!!!!!")
    return inds_not_present, map_between

# this runs the function main as the main program... this is a hack that allows functions
# to come after the main code so it presents in a more logical, C-like, way
if __name__ == '__main__':
    main()
