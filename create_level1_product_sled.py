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
# To create a uniform "level1" data product for the SPLASH research sleds
#
# ############################################################################################
# level1 "slow":
#
#       Observations from all instruments averaged over 1 minute intervals. These include
#       statistics on the one minute intervals as logged at the stations. These level1 files
#       are raw uncalibrated data.
# 
# level1 "fast":
# 
#       Observations from metek sonic anemometers. These observations
#       are taken at 20Hz barring any problems but written out at even 10Hz
#       intervals to allow for timestamping and interpretation. For more documentation
#       on the timing of these observations and how this was
#       processed, please see the comments in the "get_fast_data()"
#       function. These level1 files are raw uncalibrated data.
#
# HOWTO:
#
# To run this package with verbose printing over all ofthe data:
# python3 create_level1_product_sled.py --verbose --start_time 20210601 --end_time 202????? -p ./data/
#
# or with fancy logging and time tracking for asfs30 only:
#
# nice -n 20 time python3 -u ./create_level1_product_sled.py -v -s 20220601 -e 20220901 -pd ./ramdisk/ -a asfs30 | tee sled_(date +%F).log
#
# RELEASE-NOTES:
#
# Please look at CHANGELOG.md for notes on changes/improvements with each version.
#
# ###############################################################################################
#
# look at these files for documentation on netcdf vars and nomenclature
# as well as additional information on the syntax used to designate certain variables here
from sled_data_definitions import define_global_atts, get_level1_col_headers, get_level1_soil_col_headers
from sled_data_definitions import define_level1_slow, define_level1_fast, define_turb_variables, define_level1_soil

import functions_library as fl

import os, inspect, argparse, time, glob, sys
sys.stdout.reconfigure(line_buffering=True) # only line buffering when run as script, so we can have tee
 
import numpy  as np
import pandas as pd
nan = np.nan
pd.options.mode.use_inf_as_na = True # no inf values anywhere, inf=nan
 
import socket

global nthreads 
hostname = socket.gethostname()
if '.psd.' in hostname:
    if hostname.split('.')[0] in ['linux1024', 'linux512']:
        nthreads = 25  # the twins have 32 cores/64 threads, won't hurt if we use <30 threads
    elif hostname.split('.')[0] in ['linux64', 'linux128', 'linux256']:
        nthreads = 12  # 
    else:
        nthreads = 60  # the new compute is hefty.... real hefty

else: nthreads = 8     # laptops don't tend to have 12  cores... yet
 
from multiprocessing import Process as P
from multiprocessing import Queue   as Q

# need to debug something? kills multithreading to step through function calls 
debug = False
if debug: 
    nthreads = 1
    from multiprocessing.dummy import Process as P
    from multiprocessing.dummy import Queue   as Q
    #from debug_functions import drop_me as dm
  
from datetime  import datetime, timedelta
from numpy     import sqrt
from netCDF4   import Dataset, MFDataset

import warnings; warnings.filterwarnings(action='ignore') # vm python version problems, cleans output....

# just in case... avoids some netcdf nonsense involving the default file locking across mounts
os.environ['HDF5_USE_FILE_LOCKING']='FALSE' # just in case
os.environ['HDF5_MPI_OPT_TYPES']='TRUE'     # just in case

version_msg = '\n\n2021-2023 SPLASH sled processing code v.'+code_version[0]\
              +', last updates: '+code_version[1]+' by '+code_version[2]+'\n\n'

def printline(startline='',endline=''): # make for pretty printing
    print('{}--------------------------------------------------------------------------------------------{}'
          .format(startline, endline))

printline()
print(version_msg)

def main(): # the main data crunching program

    sled_dict = {'asfs50': 'asfs50-picnic', 'asfs30': 'asfs30-pond'} # file names

    # the UNIX epoch... provides a common reference, used with base_time
    global epoch_time
    epoch_time        = datetime(1970,1,1,0,0,0) # Unix epoch, sets time integers

    global verboseprint  # defines a function that prints only if -v is used when running
    global printline     # prints a line out of dashes, pretty boring
    global verbose       # a useable flag to allow subroutines etc when using -v 

    global data_dir, level1_dir
    global file_prefix
    file_prefix = 'ASFS_30'

    global integ_time_turb_flux
    integ_time_turb_flux = [10, 30, 60]                  # [minutes] the integration time for the turbulent flux calculation
    calc_fluxes          = False                  # if you want to run turbulent flux calculations and write files

    global nan, def_fill_int, def_fill_flt # make using nans look better
    nan = np.NaN
    def_fill_int = -9999
    def_fill_flt = -9999.0

    Rd       = 287     # gas constant for dry air
    K_offset = 273.15  # convert C to K
    h2o_mass = 18      # are the obvious things...
    co2_mass = 44      # ... ever obvious?
    sb       = 5.67e-8 # stefan-boltzmann
    emis     = 0.985   # snow emis assumption following Andreas, Persson, Miller, Warren and so on

    # there are two command line options that effect processing, the start and end date...
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_time', metavar='str', help='beginning of processing period, Ymd syntax')
    parser.add_argument('-e', '--end_time', metavar='str', help='end  of processing period, Ymd syntax')
    parser.add_argument('-v', '--verbose', action ='count', help='print verbose log messages')
    parser.add_argument('-p', '--path', metavar='str', help='base path to data up to, including /data/, include trailing slash') 
    parser.add_argument('-pd', '--pickledir', metavar='str',help='want to store a pickle of the data for debugging?')
    parser.add_argument('-nf', '--no_fluxes', metavar='str',help='dont run turbulent flux calculations, just write fast/slow')
    parser.add_argument('-a', '--station', metavar='str',help='asfs#0, if omitted all will be procesed')

    # add verboseprint function for extra info using verbose flag, ignore these 5 lines if you want
    args         = parser.parse_args()
    verbose      = True if args.verbose else False # use this to run segments of code via v/verbose flag
    v_print      = print if verbose else lambda *a, **k: None     # placeholder
    verboseprint = v_print # use this function to print a line if the -v/--verbose flag is provided

    if args.station: flux_station = args.station
    else: 
        flux_station = 'asfs30'

    global sled_name
    sled_name = sled_dict[flux_station]

    if args.path: data_dir = args.path+f'/{flux_station}'
    else: data_dir = f'/PSL/Observations/Campaigns/SPLASH/{flux_station}'
    
    level1_dir =  f'{data_dir}/1_level_ingest/'  # where does level1 data go

    global start_time, end_time, day_delta
    day_delta  = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00

    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y%m%d')
    else:
        # make the data processing start yesterday! i.e. process only most recent full day of data
        start_time = epoch_time.today() # any datetime object can provide current time
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, day=start_time.day)

    if args.end_time:
        end_time = datetime.strptime(args.end_time, '%Y%m%d')
        if end_time == start_time: end_time = start_time+day_delta 
    else:
        end_time = start_time+timedelta(1)

    if args.no_fluxes:
        calc_fluxes = False

    if args.pickledir: pickle_dir=args.pickledir
    else: pickle_dir=False

    printline(endline="\n")
    print('The first day we process data is:      %s' % str(start_time))
    print('The last day we will process data is:  %s' % str(end_time))
    printline("\n")
    print("Getting data from raw files ... and doing it in threads, hold onto your britches")
    printline()

    slow_atts, slow_vars = define_level1_slow()
    slow_data = get_slow_data(flux_station, start_time-timedelta(1), end_time+timedelta(1), data_dir)
    
    soil_atts, soil_vars = define_level1_soil()
    soil_data = get_soil_data(flux_station, start_time-timedelta(1), end_time+timedelta(1), data_dir)
        
    verboseprint("\n===================================================")
    verboseprint(f"Data and observations provided by {sled_name}:")
    verboseprint('===================================================')
    if verbose: 
        slow_data.info(verbose=True) # must be contained;
        printline()

    print("\n... got all slow data from files, exiting for timing")

    # tell user about big data gaps, good sanity check for downtime
    # using batt_volt_Avg because if the battery voltage is missing....
    if verbose: 
        
        bv = slow_data["batt_volt_Avg"]

        threshold  = 10 # warn if the station was down for more than 60 minutes
        nan_groups = bv.isnull().astype(int).groupby(bv.notnull().astype(int).cumsum()).cumsum()
        mins_down  = np.sum( nan_groups > 0 )

        prev_val   = 0 
        if np.sum(nan_groups > threshold) > 0:
            print(f"\n!!! {sled_name} was down for at least {threshold} minutes on the following dates: ")
            for i,val in enumerate(nan_groups):
                if val == 1 and prev_val == 0: 
                    down_start = bv.index[i]
                if val == 0  and prev_val > 0 and prev_val > threshold: 
                    print("---> from {} to {} ".format(down_start, bv.index[i]))
                prev_val = val

            if prev_val > 0 and prev_val > threshold: 
                print(f"!!! -------------> from {down_start} to {bv.index[i]} !!!")

            perc_down = round(((bv.size-mins_down)/bv.size)*100, 3)
            print("\n... the station was down for a total of {} minutes...".format(mins_down))
            print("... giving an uptime of approx {}% over this period\n".format(perc_down))

        else:
            print(f"\n ... station {sled_name} was alive for the entire time range you requested!! Not bad... \n")

    # we'll process one day at a time, as defined here
    def process_day(today, slow_data_today, soil_data_today): 

        tomorrow = today+day_delta

        hour_q_dict = {}
        hour_p_dict = {} 

        # run the threads for retreiving data, include buffer for boundary
        fast_data_today = get_fast_data(flux_station, today-timedelta(1), today+timedelta(1)+day_delta, data_dir) 
        fdt = fast_data_today[today-timedelta(hours=1):today+day_delta+timedelta(hours=1)]  
        sdt = slow_data_today[today-timedelta(hours=1):today+day_delta+timedelta(hours=1)]  
        pdt = soil_data_today[today-timedelta(hours=1):today+day_delta+timedelta(hours=1)]  

        satts = slow_atts
        fatts = fast_atts
        patts = soil_atts

        # these aren't written to disk (not in definitions files) as of right now
        Td, h, a, x, Pw, Pws, rhi = fl.calc_humidity_ptu300(sdt['vaisala_RH_Avg'],\
                                                            sdt['vaisala_T_Avg']+K_offset,
                                                            sdt['vaisala_P_Avg'],
                                                            0)
        sdt['rhi']                  = rhi 
        sdt['abs_humidity_vaisala'] = a
        sdt['vapor_pressure']       = Pw
        sdt['mixing_ratio']         = x
        sdt['skin_temp_surface']    = (((sdt['ir20_lwu_Wm2_Avg']-(1-emis)*sdt['ir20_lwd_Wm2_Avg'])/(emis*sb))**0.25)-K_offset
        
        # fl.pickle_function_args_for_debugging((sdt, today),
        #                                       './pickles/', f'{today.strftime("%Y%m%d")}_slowdebug_pre.pkl')


        if calc_fluxes and not fdt.empty:

            # let's thread the flux QC/calculations for every hour because it's very slow and 24/6=4, that's nice
            hours_today = pd.date_range(today, tomorrow, freq='H'); 
            hour_delta  = pd.to_timedelta(3599999999,unit='us') # up to but not including the hour

            qc_arg_list = [] 
            for hour in hours_today: qc_arg_list.append((fdt[hour-timedelta(hours=1):hour+hour_delta+timedelta(hours=1)].copy(),
                                                         sdt[hour-timedelta(hours=1):hour+hour_delta+timedelta(hours=1)].copy(),
                                                         hour-timedelta(hours=1), hour+hour_delta+timedelta(hours=1),))

            frame_list = call_function_threaded(qc_fast_data, qc_arg_list) # actually call qc func for every hour
            qced_10hz = pd.concat(frame_list).sort_index().drop_duplicates() # concat the returned hourly dataframes into one list again
            calc_arg_list = []
            for hour in hours_today: 
                if not qced_10hz[hour:hour+hour_delta].empty:
                    calc_arg_list.append((qced_10hz[hour-timedelta(hours=1):hour+hour_delta+timedelta(hours=1)],
                                          sdt[hour-timedelta(hours=1):hour+hour_delta+timedelta(hours=1)],
                                          hour, hour+hour_delta,))

            # pickle_name = f'{hours_today[0].strftime("%Y%m%d")}_{hours_today[-1].strftime("%Y%M%d")}rotate_and_fluxulate.pkl
            # pickle_function_args_for_debugging(calc_arg_list, pickle_dir, pickle_name)
            dict_list = call_function_threaded(rotate_and_fluxulate, calc_arg_list) # calc fluxes now 

            tw_arg_list = [];
            for integration_window in integ_time_turb_flux: 
                turb_df_list = [] # each hour
                if integration_window == integ_time_turb_flux[0]: slow_df_list = [] # each hour
                for turbd, slow_data in dict_list:
                    turb_df_list.append(turbd[integration_window])
                    if integration_window == integ_time_turb_flux[0]: slow_df_list.append(slow_data)
                if len(turb_df_list) != 0: tw_arg_list.append((pd.concat(turb_df_list), sled_name, today, integration_window,))
                else: print(f"!!! there were ZERO turbulence values on {today}, was there data???")

            try: 
                sdd = pd.concat(slow_df_list).sort_index()
                sdd = sdd.drop_duplicates() 
                sdt = sdd[~sdd.index.duplicated('first')]
            except ValueError: 
                print(f"!!! was there no slow data today ???")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n {sdt[today:tomorrow]['batt_volt_Avg']}\n\n")
                print(f"!!! missing points: {sdt[today:tomorrow]['batt_volt_Avg'].isna().sum()} !!!\n")

            # fl.pickle_function_args_for_debugging((sdt, slow_df_list),
            #                                       './pickles/', f'{today.strftime("%Y%m%d")}_slowdebug_post.pkl')
            trash_list = call_function_threaded(write_turb_netcdf, tw_arg_list) # calc fluxes now 

        elif fdt.empty:
            print(f"!!! No fast  data {today}, not calculating turb fluxes !!!")

        # first we prioritize writing the fast and slow files, then we calculate turbulent fluxes
        if not sdt.empty or not fdt.empty or not pdt.empty: 
            # run the threads for retreiving data
            fast_q = Q()
            slow_q = Q()
            soil_q = Q()
            p_fast = P(target=write_level1_fast, args=(fdt[today:tomorrow].copy(), fatts, today, fast_q))
            p_slow = P(target=write_level1_slow, args=(sdt.loc[today:tomorrow].copy(), satts, today, slow_q))
            p_soil = P(target=write_level1_soil, args=(pdt.loc[today:tomorrow].copy(), patts, today, soil_q))
            if nthreads == 1:
                p_fast.start(); trash_fast = fast_q.get(); p_fast.join(timeout=1.0)
                p_slow.start(); trash_slow = slow_q.get(); p_slow.join(timeout=1.0)
                p_soil.start(); trash_soil = soil_q.get(); p_soil.join(timeout=1.0)
            else:
                p_fast.start()            ; p_slow.start(); p_soil.start() 
                trash_slow = slow_q.get() ; trash_fast = fast_q.get() ; trash_soil = soil_q.get() 
                p_slow.join(timeout=1.0)  ; p_fast.join(timeout=1.0)  ; p_soil.join(timeout=1.0) 

        else:
            print("... no data available for {} on {}".format(sled_name, today))
            return 

    # #################################################################################################
    # actually call processing for the requested time range, not threadded because turbulence calcs are't
    fast_atts, fast_cols = define_level1_fast()    
    day_series = pd.date_range(start_time, end_time) # data was requested for these days

    printline(endline=f"\n\n  Processing all requested days of data for {sled_name}\n"); printline()
    for iday, curr_day in enumerate(day_series):
        process_day(curr_day, slow_data[curr_day-timedelta(hours=1):curr_day+day_delta+timedelta(hours=1)],soil_data[curr_day-timedelta(hours=1):curr_day+day_delta+timedelta(hours=1)])

    printline()
    print('All done! Netcdf output files can be found in: {}'.format(level1_dir))
    printline()
    print(version_msg)
    printline()

# this abstracts the threading so there's not a bunch of copied code. this is pretty stupid threading...
# ... it only allows for one "q.put" and "q.get" at the moment, meaning one return value. code accordingly
# it also assumes that q is the last argument of the function being called. 

# this is so ugly. this needs to be replaced with pools and handle exceptions in a reasonable way.
# it would be *better* (less error prone) cleaner
def call_function_threaded(func_name, arg_list):
    q_list = []
    p_list = []
    ret_list = [] 
    for i_call, arg_tuple in enumerate(arg_list):
        q = Q(); q_list.append(q)
        p = P(target=func_name, args=arg_tuple+(q,))
        p_list.append(p); p.start()
        if (i_call+1) % nthreads == 0 or arg_tuple is arg_list[-1]:
            for iq, fq in enumerate(q_list):
                rval = fq.get()
                p_list[iq].join(timeout=1.0)
                ret_list.append(rval)
            q_list = []
            p_list = []

    return ret_list

def get_file_list(station_name, first_time, last_time, file_type, data_dir):

    prefix_dict = {'asfs50':'ASFS_50', 'asfs30':'ASFS_30'}
    
    filename_list = [] # list of filenames to return
    date_list     = []
    stream = 'card'
    if stream == 'card':
        file_prefix   = prefix_dict[station_name]; time_format = '%Y%m%d%H'; time_format_short = '%Y%m%d%H'
    else:
        file_prefix = 'IGB-H-002'; time_format = '%Y_%m_%d_%H_%M_%S'; time_format_short = '%Y_%m_%d_%H'

    if stream == 'card': time_window = pd.to_timedelta(0 ,unit='d') # 
    else: time_window = pd.to_timedelta(3 ,unit='d') # look for files in this window
    hour_delta    = pd.to_timedelta(60 ,unit='m') 
    curr_time     = first_time.replace(minute=0, second=0, microsecond=0)-hour_delta - time_window
    data_dir = data_dir+ f'/0_level_raw/daily_files/'

    # if all files are in one directory (i.e. onboard processing), you need to delete this outer loop... it makes
    # things super slow also.. this could be way more clever with some assumptions about directory structure
    for subdir in [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]:
        curr_time = first_time.replace(minute=0, second=0, microsecond=0)-hour_delta - time_window
        while curr_time <= last_time.replace(minute=0, second=0, microsecond=0)+hour_delta + time_window: 
        
            for f in glob.glob(f"{data_dir}/{subdir}/{file_prefix}_{file_type}_{curr_time.strftime(time_format_short)}*.dat"):
                if os.path.exists(f):
                    if stream=='card': tstr = f.split('.')[-2].split('_')[-1][0:-2]
                    else: tstr = f.split('.')[-2].split('slow_')[-1]
                    try: t = datetime.strptime(tstr, time_format)
                    except:
                        print(f"The following file could not be time parsed: {tstr} {f}")
                        continue

                    filename_list.append(f); date_list.append(t)
                    if len(filename_list) %100 == 0: print(f"... found {len(filename_list)} files so far...")
            curr_time = curr_time+hour_delta

    return filename_list, date_list

# pulls in logger text files from sdcard data and returns a dataframe and a list of data attributes (units, etc)
# qq returns the results from a threaded call and is of Queue class
def get_slow_data(station_name, start_time, end_time, data_dir): 

    filename_list, date_list = get_file_list(station_name, start_time, end_time, 'slow', data_dir)
    print(f'... found {len(filename_list)} slow files in directory {data_dir}')
    
    frame_list = [] # list of frames from files... to be concatted after loop
    data_atts, data_cols  = define_level1_slow()

    # called in process/queue loop over card_file_list below
    def read_slow_csv(dfile, file_q):
        # 'cause campbell is.... inconsistent
        na_vals = ['nan','NaN','NAN','NA','\"INF\"','\"-INF\"','\"NAN\"','\"NA','\"NAN','inf','-inf''\"inf\"','\"-inf\"','']

        # count the columns in the first line and select the col headers out of asfs_data_definitions.py
        with open(dfile) as ff: 
            topline = ff.readline().rsplit(',')
            num_cols = len(topline)
            if num_cols > 8: # you dont have a header
                header_rows = []
            if num_cols == 8: # then the file has a header
                num_cols = len(ff.readline().rsplit(',')) # so get the right number from the next line
                header_rows = [r for r in range(0,4)]

        if num_cols == 97: 
            cver = 0
        else: 
            print(f"!!! something went really wrong with the number of columns in the file, can't use!!! {num_cols} {dfile}")
            file_q.put(pd.DataFrame()); return pd.DataFrame()

        cols  = get_level1_col_headers(num_cols, cver)
        try: 
            frame = pd.read_csv(dfile, parse_dates=[0], sep=',', na_values=na_vals,\
                                index_col=0, engine='c', names=cols, skiprows=header_rows)
            file_q.put(frame); return frame
        except Exception as e:  
            printline()
            print(e)
            print("!!! there was an exception reading file {}, {} skipping!!!".format(data_file, num_cols))
            printline()
            file_q.put(pd.DataFrame()); return pd.DataFrame()

    print("... getting slow files into DataFrames (threaded)...")
    arg_list = []
    for file_i, data_file in enumerate(filename_list):
        arg_list.append((data_file,)) 
    frame_list = call_function_threaded(read_slow_csv, arg_list)

    print(f"... got all files read, got all threads back, now we concat")
    try:
        comb_frame = pd.concat(frame_list) # is concat computationally efficient?    
        comb_frame = comb_frame.sort_index() # sort chronologically    
        data_frame = comb_frame.drop_duplicates() # get rid of duplicates
        data_frame = data_frame.groupby(data_frame.index).last() # another edge case
        before_len = len(comb_frame.index)
        after_len  = len(data_frame.index)
    except:
        print("!!! there was no data for today it would seem... \n ... FAILING UNGRACEFULLY"); raise

    # subtract 1 min from the index so time marks beginning rather than end of the 1 min avg time, more conventional-like
    data_frame.index = data_frame.index-pd.Timedelta(1,unit='m')  

    data_frame = data_frame[start_time:end_time] # done gathering data, now narrow down to requested window

    print(f"... dropped {before_len-after_len} duplicates from slow data!")

    # sanity check
    if data_frame.empty: 
        #print("No {} data for requested time range {} ---> {} ?\n".format(searchdir,start_time,end_time))
        print("... Im sorry... no slow data will be written for {}".format(station))
        return data_frame

    # now, reindex for every minute in range and fill with nans so that the we commplete record
    mins_range = pd.date_range(start_time, end_time-pd.to_timedelta(1,unit='us'), freq='T') # all the minutes today, for obs
    try:
        data_frame = data_frame.reindex(labels=mins_range, copy=False)
    except Exception as ee: # this shouldn't ever happen
        printline()
        #print("There was an exception reindexing for {}".format(searchdir))
        print(' ---> {}'.format(ee))
        printline()
        print(data_frame.index[data_frame.index.duplicated()])
        print("... this shouldn't happen but trying to recover, dropping these duplicate indexes... ")
        printline()
        data_frame = data_frame.drop(data_frame.index[data_frame.index.duplicated()], axis=0)
        print(" There are now {} duplicates in DF".format(len(data_frame[data_frame.index.duplicated()])))
        print(" There are now {} duplicates in timeseries".format(len(mins_range[mins_range.duplicated()])))
        data_frame = data_frame.reindex(labels=mins_range, copy=False)

    # why doesn't the dataframe constructor use the name from cols for the index name??? silly
    data_frame.index.name = data_cols[0]


    # sort data by time index and return data_frame to queue
    return data_frame
    

# pulls in logger text files from sdcard data and returns a dataframe and a list of data attributes (units, etc)
# qq returns the results from a threaded call and is of Queue class
def get_soil_data(station_name, start_time, end_time, data_dir): 

    filename_list, date_list = get_file_list(station_name, start_time, end_time, 'soil', data_dir)
    print(f'... found {len(filename_list)} soil files in directory {data_dir}')
    
    frame_list = [] # list of frames from files... to be concatted after loop
    data_atts, data_cols  = define_level1_soil()

    # called in process/queue loop over card_file_list below
    def read_soil_csv(dfile, file_q):
        # 'cause campbell is.... inconsistent
        na_vals = ['nan','NaN','NAN','NA','\"INF\"','\"-INF\"','\"NAN\"','\"NA','\"NAN','inf','-inf''\"inf\"','\"-inf\"','']

        # count the columns in the first line and select the col headers out of asfs_data_definitions.py
        with open(dfile) as ff: 
            topline = ff.readline().rsplit(',')
            num_cols = len(topline)
            if num_cols > 8: # you dont have a header
                header_rows = []
            if num_cols == 8: # then the file has a header
                num_cols = len(ff.readline().rsplit(',')) # so get the right number from the next line
                header_rows = [r for r in range(0,4)]
        
        if num_cols == 25: 
            cver = 0
        else: 
            print(f"!!! something went really wrong with the number of columns in the file, can't use!!! {num_cols} {dfile}")
            file_q.put(pd.DataFrame()); return pd.DataFrame()

        cols  = get_level1_soil_col_headers(num_cols, cver)
        try: 
            frame = pd.read_csv(dfile, parse_dates=[0], sep=',', na_values=na_vals,\
                                index_col=0, engine='c', names=cols, skiprows=header_rows)
            file_q.put(frame); return frame
        except Exception as e:  
            printline()
            print(e)
            print("!!! there was an exception reading file {}, {} skipping!!!".format(data_file, num_cols))
            printline()
            file_q.put(pd.DataFrame()); return pd.DataFrame()

    print("... getting soil files into DataFrames (threaded)...")
    arg_list = []
    for file_i, data_file in enumerate(filename_list):
        arg_list.append((data_file,)) 
    frame_list = call_function_threaded(read_soil_csv, arg_list)

    print(f"... got all files read, got all threads back, now we concat")
    try:
        comb_frame = pd.concat(frame_list) # is concat computationally efficient?    
        comb_frame = comb_frame.sort_index() # sort chronologically    
        data_frame = comb_frame.drop_duplicates() # get rid of duplicates
        before_len = len(comb_frame.index)
        after_len  = len(data_frame.index)
    except:
        print("!!! there was no data for today it would seem... \n ... FAILING UNGRACEFULLY"); raise

    data_frame = data_frame[start_time:end_time] # done gathering data, now narrow down to requested window

    print(f"... dropped {before_len-after_len} duplicates from soil data!")

    # sanity check
    if data_frame.empty: 
        #print("No {} data for requested time range {} ---> {} ?\n".format(searchdir,start_time,end_time))
        print("... Im sorry... no soil data will be written for {}".format(station))
        return data_frame

    # why doesn't the dataframe constructor use the name from cols for the index name??? silly
    data_frame.index.name = data_cols[0]


    # sort data by time index and return data_frame to queue
    return data_frame


# gets 10hz data from metek sonics, 
def get_fast_data(station_name, first_time, last_time, data_dir): 

    file_prefix = station_name.upper()

    # get data cols/attributes from definitions file
    data_atts, data_cols = define_level1_fast()

    eighth_day_delta = pd.to_timedelta(10800000000 ,unit='us') # we want to eliminate interpolation boundary conds
    #half_day_delta  = pd.to_timedelta(43200000000 ,unit='us') # we want to eliminate interpolation boundary conds
    day_delta        = pd.to_timedelta(86399999999,unit='us')  # we want to go up to but not including 00:00

    filename_list, date_list = get_file_list(station_name, first_time, last_time, 'fast', data_dir)

    def read_fast_csv(file_name, file_date, file_q):
        na_vals = ['-9999', '-9999.0','nan','NaN','NAN','NA','\"INF\"','\"-INF\"','\"NAN\"',
                   '\"NA','\"NAN','inf','-inf''\"inf\"','\"-inf\"','']

        # count the columns in the first line, if less than expected, bail
        with open(file_name) as ff: 

            topline = ff.readline().rsplit(',')
            num_cols = len(topline)
            header_rows = []
            if num_cols == 8: # then the file has a header
                num_cols = len(ff.readline().rsplit(',')) # so get the right number from the next line
                header_rows = [r for r in range(0,4)]
            if num_cols == 12:
                data_cols.insert(1,"LINE")            

        try:  # ingest each csv into own data frame and keep in list
            frame = pd.read_csv(file_name, parse_dates=[0], sep=',', na_values=na_vals,\
                                engine='c', names=data_cols, skiprows=header_rows)
            file_q.put((frame, file_date))

        except Exception as e:
            printline()
            print(e)
            print("!!! There was an exception reading file {}, {} skipping!!!".format(file_name, num_cols))
            printline()
            file_q.put((pd.DataFrame(), file_date))

    print("... getting fast files into DataFrames (threaded)...")
    arg_list = []
    for file_i, data_file in enumerate(filename_list):
        arg_list.append((data_file, date_list[file_i])) 

    ret_list    = call_function_threaded(read_fast_csv, arg_list)
    frame_list  = list((x[0] for x in ret_list))
    frame_dates = list((x[1] for x in ret_list))

    # we have to order frames by date before we reindex
    sorted_indexes = np.argsort(frame_dates)
    frame_dates    = [frame_dates[i] for i in sorted_indexes]
    frame_list     = [frame_list[i] for i in sorted_indexes]

    # reindex each frame to avoid collisions before concatting/interpolating the time column/axis
    reindex_start = 0 
    for iframe, curr_frame in enumerate(frame_list): # reindex each data frame properly to avoid collisions 
        curr_frame.index = range(reindex_start, reindex_start+len(curr_frame.index))
        reindex_start    = reindex_start + len(curr_frame.index) # increment by num data rows for next loop

    if len(frame_list) == 0:
        print(f"!!! no fast data in range {first_time} ---> {last_time} !!!")
        return pd.DataFrame()

    catted_frame = pd.concat(frame_list)
    catted_frame = catted_frame.sort_index()
    catted_frame = catted_frame.drop_duplicates() 

    # this drop_duplicates is complicated because of duplicates between the shunted radio files and sd cards
    sub_frame     = catted_frame[data_cols[1:7]]
    dupes         = sub_frame.duplicated() # we want to keep the second half of the duplicates, no tilde, I told you it was weird1
    deduped_frame = catted_frame[dupes == False]
    full_frame    = deduped_frame.copy()

    # this fixes the fast timestamping in 5 second blocks issues...
    # ... it's way slower than the old interpolation, but much more correct
    def correct_timestamps(df, q=None):

        df.index = range(0, len(df)) # have to give it a simple index to count points with

        # use 'duplicated' to mark the start of the next scan cycle
        times     = df["TIMESTAMP"].copy()
        new_times = times.copy()
        dupes     = times[times.duplicated()==False]
        ind_iter  = iter(dupes.index)

        tot_len = len(df)

        try: curr_i = next(ind_iter)
        except: 
            not_do = "true-true" # this condition should never be hit

        # loop over each block of equal timestamps and interpolate only between blocks
        break_me = False
        while True:
            try: next_i = next(ind_iter)
            except:
                break_me = True
                next_i = tot_len

            dist = next_i-curr_i
            if dist == 0: 
                print(f"... weirdness {curr_i}, this should never happen")
                break #weird

            tstep = 10/dist # supposed to be 10 second timesteps
            tdelt = timedelta(0,tstep)

            successes=0
            for ii in range(curr_i, next_i):
                new_times.at[ii] = times.loc[ii]+(ii-curr_i)*tdelt
                successes+=1


            if break_me:
                break
            curr_i  = next_i

        # well. not quite done. now we subtract 10 sec (length of the scan) from the index so that
        # the times mark the beginning rather than the end
        offset_delta  = pd.to_timedelta(10, unit='s') 

        new_times = new_times - offset_delta
        df['TIMESTAMP'] = new_times
        df.index = new_times

        try: q.put(df)
        except: return df

    # some hacky code to break this into chunks because it's super slow
    times = full_frame["TIMESTAMP"].copy()
    tdupes = times[times.duplicated()==False]

    print(f"... correcting fast timestamps  (threaded but still really slow)... {first_time}-{last_time} ")
    arg_list = []
    start_i  = full_frame.index[0]
    skip_tot = int(len(tdupes)/nthreads)

    for t in range(0,nthreads):
        if t == nthreads-1: 
            end_i = tdupes.index[-1]-1
            next_i = full_frame.index[-1]
        else: 
            end_i = skip_tot*(t+1)
            next_i = tdupes.index[end_i]-1

        partial_frame = full_frame.loc[start_i:next_i]
        if not partial_frame.empty: arg_list.append( (partial_frame,) ) # comma, function args must be tuple
        start_i = next_i+1
        
    try: fixed_frame_list = call_function_threaded(correct_timestamps, arg_list)
    except Exception as e: fixed_frame_list = []

    full_frame       = pd.concat(fixed_frame_list)
    idf              = pd.DataFrame(full_frame.index) # debug helper, not used

    # full_frame contains all the data now, in the correct order, but times are wonky and
    # it's indexed by an incremented integer, not so useful. fix it
    return_frame = full_frame.sort_index()
    return_frame = return_frame[first_time:last_time-pd.to_timedelta(1,'us')]
    return_frame = return_frame.drop_duplicates()
    n_entries    = return_frame.index.size
    
    if any(return_frame.duplicated()): 
        raise Exception

    # print("... {}/1728000 fast entries on {}-->{}, representing {}% data coverage"
    #       .format(n_entries, first_time, last_time , round((n_entries/1728000)*100.,2)))

    return return_frame


# clean up fast data for a specified time range, hourly above.. but could be daily
# licor variables kept in here, so that this code can be easily used in "standard" configuration
def qc_fast_data(fast_data_20hz, slow_data, first_time, last_time, qc_q=None):

    fd = fast_data_20hz

    # all the 0.1 seconds in time range 
    Hz10_range        = pd.date_range(first_time, last_time, freq='0.1S') 
    seconds_range     = pd.date_range(first_time, last_time, freq='S')    # all the seconds in time range, for obs
    minutes_range     = pd.date_range(first_time, last_time, freq='T')    # all the minutes in time range, for obs
    ten_minutes_range = pd.date_range(first_time, last_time, freq='10T')  # all the 10 minutes in time range, for obs

    #                              !! Important !!
    #   first resample to 10 Hz by averaging and reindexed to a continuous 10 Hz time grid (NaN at
    #   blackouts) of Lenth 60 min x 60 sec x 10 Hz = 36000 Later on (below) we will fill all missing
    #   times with the median of the (30 min?) flux sample.
    # ~~~~~~~~~~~~~~~~~~~~~ (2) Quality control ~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"... quality controlling the fast data now for {first_time}")

    # check to see if fast data actually exists...
    # no data for param, sometimes fast is missing but slow isn't... very rare
    fast_var_list = ['metek_x', 'metek_y','metek_z','metek_T', 'metek_heatstatus',
                     'licor_h2o','licor_co2','licor_pr','licor_co2_str','licor_diag']

    for param in fast_var_list:
        try: test = fd[param]
        except KeyError: fd[param] = nan

    if fd.empty: # create a fake dataframe
        nan_df = pd.DataFrame([[nan]*len(fd.columns)], columns=fd.columns)
        nan_df = nan_df.reindex(pd.DatetimeIndex([first_time]))
        fd = nan_df.copy()

    # get rid of large numbers of sequential zeros at this threshold
    consec_zero_thresh = 50 #? is this a smart threshold?
    def zero_runs(arr):
        iszero  = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges  = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges # returns [x][2] array where [2] are pairs of fu

    metek_list = ['metek_x', 'metek_y','metek_z','metek_T']
    for param in metek_list: # identify when T==0 is actually missing data, this takes some logic
        zero_ranges = zero_runs(fd[param].values)
        for seq in zero_ranges:
            if seq[1]-seq[0] > consec_zero_thresh: fd[param].iloc[seq[0]:seq[1]] = nan
    
    T_thresh          = (-70  ,40)       # minimum & maximum air temperatures (C)
    ws_thresh         = (0    ,40)       # wind speed from sonics (m/s)
    lic_co2sig_thresh = (94   ,105)      # rough estimate of minimum CO2 signal value corresponding to
                                         # optically-clean window. < 90 = needs cleaned (e.g., salt residue); < ~80?? ~ ice!
    p_thresh          = (700  ,1100)     # air pressure
    lic_h2o           = (1e-3 ,50)       # Licor h2o [mg/m3]
    lic_co2           = (300  ,5000)     # Licor co2 [g/m3]
    max_bad_paths     = (0.01 ,1)        # METEK: maximum [fraction] of bad paths allowed. (0.01 == 1%), but is
                                         # actually in 1/9 increments. This just says we require all paths to be usable.

    # begin with bounding the Metek data to the physically-possible limits
    fd ['metek_T']  [fd['metek_T'] < T_thresh[0]]           = nan
    fd ['metek_T']  [fd['metek_T'] > T_thresh[1]]           = nan
    fd ['metek_x']  [np.abs(fd['metek_x'])  > ws_thresh[1]] = nan
    fd ['metek_y']  [np.abs(fd['metek_y'])  > ws_thresh[1]] = nan
    fd ['metek_z']  [np.abs(fd['metek_z'])  > ws_thresh[1]] = nan

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
    status = fd['metek_heatstatus']
    bad_data = (status/1000-np.floor(status/1000)) >  max_bad_paths[0]
    fd['metek_x'][bad_data]=nan
    fd['metek_y'][bad_data]=nan
    fd['metek_z'][bad_data]=nan
    fd['metek_T'][bad_data]=nan

    # Physically-possible limits
    fd['licor_h2o'] .mask( (fd['licor_h2o']<lic_h2o[0]) | (fd['licor_h2o']>lic_h2o[1]) , inplace=True) # ppl
    fd['licor_co2'] .mask( (fd['licor_co2']<lic_co2[0]) | (fd['licor_co2']>lic_co2[1]) , inplace=True) # ppl
    fd['licor_pr']  .mask( (fd['licor_pr']<p_thresh[0]) | (fd['licor_pr']>p_thresh[1]) , inplace=True) # ppl

    # CO2 signal strength is a measure of window cleanliness applicable to CO2 and H2O vars
    # first map the signal strength onto the fast data since it is empty in the fast files

    # THIS WAS COMMENTED OUT, COMMENT BACK IN FOR ASFS CODE!!!!!!!!!!!!!!!!!!!
    fd['licor_co2_str'] = slow_data['licor_co2_str_out_Avg'].reindex(fd.index).interpolate()
    fd['licor_h2o'].mask( (fd['licor_co2_str']<lic_co2sig_thresh[0]), inplace=True) # ppl
    fd['licor_co2'].mask( (fd['licor_co2_str']<lic_co2sig_thresh[0]), inplace=True) # ppl

    # The diagnostic is coded                                       
    print("... decoding Licor diagnostics. It's fast like the Dranitsyn, even vectorized. Gimme a minute...")

    pll, detector_temp, chopper_temp = fl.decode_licor_diag(fd['licor_diag'])
    # Phase Lock Loop. Optical filter wheel rotating normally if 1, else "abnormal"
    bad_pll = pll == 0
    # If 0, detector temp has drifted too far from set point. Should yield a bad calibration, I think
    bad_dt = detector_temp == 0
    # Ditto for the chopper housing temp
    bad_ct = chopper_temp == 0
    # Get rid of diag QC failures
    fd['licor_h2o'][bad_pll] = nan
    fd['licor_co2'][bad_pll] = nan
    fd['licor_h2o'][bad_dt]  = nan
    fd['licor_co2'][bad_dt]  = nan
    fd['licor_h2o'][bad_ct]  = nan
    fd['licor_co2'][bad_ct]  = nan

    # Despike: meant to replace despik.m by Fairall. Works a little different tho
    #   Here screens +/-5 m/s outliers relative to a running 1 min median
    #
    #   args go like return = despike(input,oulier_threshold_in_m/s,window_length_in_n_samples)
    #
    #   !!!! Replaces failures with the median of the window !!!!

    fd['metek_x']   = fl.despike(fd['metek_x'],5,1200,'yes')
    fd['metek_y']   = fl.despike(fd['metek_y'],5,1200,'yes')
    fd['metek_z']   = fl.despike(fd['metek_z'],5,1200,'yes')
    fd['metek_T']   = fl.despike(fd['metek_T'],5,1200,'yes')           
    fd['licor_h2o'] = fl.despike(fd['licor_h2o'],0.5,1200,'yes')
    fd['licor_co2'] = fl.despike(fd['licor_co2'],50,1200,'yes')

    # ~~~~~~~~~~~~~~~~~~~~~~~ (3) Resample  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('... re-sampling to even 10 Hz grid, filled out with nans')
    #
    # 20 Hz irregular grid -> 10 Hz regular grid
    #
    # The method is to first resample the 20 Hz data to a 10 Hz regular
    # grid using the average of the (expect n=2) points at each 0.1s
    # interval. Then the result is indexed onto a complete grid for the
    # whole day, which is nominally 1 hour = 36000 samples at 10 Hz
    # Missing data (like NOAA Services blackouts) are nan
    fd_10hz = fd.resample('100ms').mean() # already at 10hz
    try: 
        print('... re-indexing to even 10 Hz grid, filled out with nans')
        fd_10hz_ri = fd_10hz.reindex(index=Hz10_range, method='nearest', tolerance='50ms')
        fd_10hz = fd_10hz_ri
    except Exception as e:
        print(f"wtf: {e}")
        dm({**locals(), **globals()}, 0)

    try: qc_q.put(fd_10hz); return fd_10hz
    except: return fd_10hz

def rotate_and_fluxulate(fast_data_10hz, slow_data, first_time, last_time, turb_q=None):

    verbose=True
    verboseprint = print
    integ_time_turb_flux = [10, 30, 60]                  # [minutes] the integration time for the turbulent flux calculation


    # ~~~~~~~~~~~~~~~~~ (4) Do the Tilt Rotation  ~~~~~~~~~~~~~~~~~~~~~~
    print(f"... cartesian tilt rotation. Translating body -> earth coordinates with {nthreads}")

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

    sd = slow_data
    fd = fast_data_10hz
   
    cd_lim            = (-2.3e-3,1.5e-2) # drag coefficinet sanity check. really it can't be < 0, but a small negative
                                         # threshold allows for empiracally defined (from EC) 3 sigma noise distributed about 0.
 
    licor_vars = ['licor_h2o', 'licor_co2', 'licor_pr']
    #for lv in licor_vars: fd[lv] = nan  # remove this when we have a licor

    inclX_interped = fl.interpolate_nans_vectorized(sd['metek_InclX_Avg'].reindex(fd.index).values)
    inclY_interped = fl.interpolate_nans_vectorized(sd['metek_InclY_Avg'].reindex(fd.index).values)
    hdg_interped   = fl.interpolate_nans_vectorized(sd['gps_hdg_Avg'].reindex(fd.index).values)
    ct_u, ct_v, ct_w = fl.tilt_rotation(inclY_interped,
                                        inclX_interped,
                                        hdg_interped,
                                        fd['metek_y'], fd['metek_x'], fd['metek_z'])

    # reassign corrected vals in meteorological convention
    fd['metek_x'] = ct_u 
    fd['metek_y'] = ct_v*-1
    fd['metek_z'] = ct_w   

    # start referring to xyz as uvw now
    fd.rename(columns={'metek_x':'metek_u'}, inplace=True)
    fd.rename(columns={'metek_y':'metek_v'}, inplace=True)
    fd.rename(columns={'metek_z':'metek_w'}, inplace=True)

    # !!
    # Now we recalculate the 1 min average wind direction and speed from the u and v velocities.
    # These values differ from the stats calcs (*_ws and *_wd) in two ways:
    #   (1) The underlying data has been quality controlled
    #   (2) We have rotated that sonic y,x,z into earth u,v,w
    #
    # I have modified the netCDF build to use *_ws_corr and *_wd_corr but have not removed the
    # original calculation because I think it is a nice opportunity for a sanity check. 
    print('... calculating a corrected set of slow wind speed and direction.')

    u_min = fd['metek_u'].resample('1T',label='left').apply(fl.take_average)
    v_min = fd['metek_v'].resample('1T',label='left').apply(fl.take_average)
    w_min = fd['metek_w'].resample('1T',label='left').apply(fl.take_average)
    ws = np.sqrt(u_min**2+v_min**2)
    wd = np.mod((np.arctan2(-u_min,-v_min)*180/np.pi),360)

    # ~~~~~~~~~~~~~~~~~~ (5) Recalculate Stats ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # !!  Sorry... This is a little messed up. The original stats are read from the NOAA Services stats
    # files, contents calculated from the raw data. But we have QC'd the data and changed the raw
    # values, so we need to update the stats. I do that here. But then was there ever any point in
    # reading the stats data in the first place?
    print('... recalculating NOAA Services style stats with corrected, rotated, and QCed values.')

    sd['wspd_vec_mean'] = ws
    sd['wdir_vec_mean'] = wd
    sd['wspd_u_mean']   = u_min
    sd['wspd_v_mean']   = v_min
    sd['wspd_w_mean']   = w_min
    sd['temp_variance_metek'] = fd['metek_T'].resample('1T',label='left').var()
    sd['acoustic_temp']       = fd['metek_T'].resample('1T',label='left').mean()

    sd['h2o_licor']            = fd['licor_h2o'].resample('1T',label='left').mean()
    sd['co2_licor']            = fd['licor_co2'].resample('1T',label='left').mean()
    sd['pr_licor']             = fd['licor_pr'].resample('1T',label='left').mean()*10 # [to hPa]

    # ~~~~~~~~~~~~~~~~~~~~ (6) Flux Capacitor  ~~~~~~~~~~~~~~~~~~~~~~~~~
    if not len(fd.index) > 1:
        print("!!! cannot calculate fluxes because there was no fast data for {first_time} ---> {last_time}")
        try: turb_q.put(None); return None
        except: return None

    else:

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

        metek_10hz = fd[['metek_u', 'metek_v', 'metek_w','metek_T']].copy()
        metek_10hz.rename(columns={\
                                   'metek_u':'u',
                                   'metek_v':'v',
                                   'metek_w':'w',
                                   'metek_T':'T'}, inplace=True)
        licor_10hz = fd[['licor_h2o', 'licor_co2']].copy()

        data_dicts = {}
        for win_num, integration_window in enumerate(integ_time_turb_flux):
            flux_freq_str   = '{}T'.format(integration_window)
            flux_time_today = pd.date_range(first_time, last_time+timedelta(microseconds=1), freq=flux_freq_str) 

            for time_i in range(0,len(flux_time_today)-1): # flux_time_today = a DatetimeIndex defined earlier and based
                                                           # on integ_time_turb_flux, the integration window for the
                                                           # calculations that is defined at the top of the code

                if time_i == 0:
                    verboseprint(f'... turbulence integration across {flux_freq_str} for {flux_time_today[time_i]}')

                # Get the index, ind, of the metek frame that pertains to the present calculation 
                # A little tricky. We need to make sure we give it enough data to encompass the nearest power of 2:
                # for 30 min fluxes this is ~27 min so you are good, but for 10 min fluxes it is 13.6 min so you need to give it more. 

                # We buffered the 10 Hz so that we can go outside the edge of "today" by up to an hour.
                # It's a bit of a formality, but for general cleanliness we are going to
                # center all fluxes to the nearest min so that e.g:
                # 12:00-12:10 is actually 11:58 through 12:12
                # 12:00-12:30 is actually 12:01 through 12:28     
                po2_len  = np.ceil(2**round(np.log2(integration_window*60*10))/10/60) # @10Hz, min needed to cover nearet po2 [minutes]
                t_win    = pd.Timedelta((po2_len-integration_window)/2,'minutes')
                metek_in = metek_10hz.loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].copy()

                # we need pressure and temperature and humidity
                Pr_time_i = sd['vaisala_P_Avg'] .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                T_time_i  = sd['vaisala_T_Avg'] .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()
                Q_time_i  = sd['mixing_ratio']  .loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].mean()/1000

                # get the licor data
                licor_data = licor_10hz.loc[flux_time_today[time_i]-t_win:flux_time_today[time_i+1]+t_win].copy()

                # make the turbulent flux calculations via Grachev module
                sonic_z       = 3.3 # what is sonic_z for the flux stations
                data = fl.grachev_fluxcapacitor(sonic_z, metek_in, licor_data, 'g/m3', 'mg/m3', Pr_time_i, T_time_i,
                                                Q_time_i, PFb0, PFb1, PFb2, rotation_flag, verbose=verbose, integration_window=integration_window)

                # sanity check on Cd. Ditch the run if it fails
                #data[:].mask( (data['Cd'] < cd_lim[0])  | (data['Cd'] > cd_lim[1]) , inplace=True) 

                # doubtless there is a better way to initialize this
                if time_i == 0: turbulencetom = data
                else: turbulencetom = turbulencetom.append(data)

            # now add the indexer datetime doohicky
            turbulencetom.index = flux_time_today[0:-1] 
            turb_cols = turbulencetom.columns

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

            # (2) if missing times were found, fill with nans of the freq length you discovered. this happens on days
            # when the instruents are turned on and also perhaps runs when missing data meant the flux_capacitor
            # returned for lack of inputs
            if f_dim_len > 0 and missing_f_dim_ind:        

                # case we have no data we need to remake a nominal fs as a filler
                if 'fs' not in locals(): 
                    fs = pd.DataFrame(np.zeros((60,1)),columns=['fs'])
                    fs = fs['fs']*nan


                for ii in range(0,len(missing_f_dim_ind)):
                    # these are the array with multiple dims...  im filling the ones that are missing with nan (of fs in
                    # the case of fs...) such that they can form a proper and square array for the netcdf
                    turbulencetom['fs'][missing_f_dim_ind[ii]] = fs
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

            data_dicts[integration_window] = turbulencetom.copy()
            
            # end of EC calculation loop, calculate the bulk 
            print(f'... calculating bulk fluxes over {integration_window} for {first_time}')
            minutes_today = pd.date_range(first_time, last_time, freq='T') # all the minutes for time range
            empty_data = np.zeros(np.size(sd['mixing_ratio'][minutes_today]))

            bulk_input = pd.DataFrame()
            bulk_input['u']  = sd['wspd_vec_mean'][minutes_today]     # wind speed (m/s)
            bulk_input['ts'] = sd['skin_temp_surface'][minutes_today] # bulk water/ice surface tempetature (degC) 
            bulk_input['t']  = sd['vaisala_T_Avg'][minutes_today]     # air temperature                    (degC) 
            bulk_input['Q']  = sd['mixing_ratio'][minutes_today]/1000 # air moisture mixing ratio          (kg/kg)
            bulk_input['zi'] = empty_data+600                         # inversion height                   (m) wild guess
            bulk_input['P']  = sd['vaisala_P_Avg'][minutes_today]     # surface pressure                   (mb)
            bulk_input['zu'] = empty_data+3.3                         # height of anemometer               (m)
            bulk_input['zt'] = empty_data+2                           # height of thermometer              (m)
            bulk_input['zq'] = empty_data+2                           # height of hygrometer               (m)      
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
                       bulk_input['zu'][ii],bulk_input['zt'][ii], bulk_input['zq'][ii]] 

                if not any(np.isnan(tmp)):
                    bulkout = fl.cor_ice_A10(tmp)
                    for hh in range(len(bulkout)):
                        # if bulkout[13] < cd_lim[0] or bulkout[13] > cd_lim[1]:  # Sanity check on Cd. Ditch the whole run if it fails
                        #     bulk[bulk.columns[hh]][ii]=nan                      # for some reason this needs to be in a loop
                        # else:
                        #     bulk[bulk.columns[hh]][ii]=bulkout[hh]
                        bulk[bulk.columns[hh]][ii]=bulkout[hh]

            data_dicts[integration_window] = pd.concat( [data_dicts[integration_window].copy(), bulk], axis=1) 

    try: turb_q.put([data_dicts, sd]); return [data_dicts, sd]
    except: return [data_dicts, sd]


# do the stuff to write out the level1 files, after finishing this it could probably just be one
# function that is called separately for fast and slow... : \ maybe refactor someday.... mhm
def write_level1_fast(fast_data, fast_atts, date, fast_q):

    time_name   = list(fast_atts.keys())[0] # are slow atts and fast atts always going to be the same?
    max_str_len = 30 # these lines are for printing prettilly

    fast_atts_iter   = fast_atts.copy() # so we can delete stuff from fast atts and not mess with iterator
    fast_atts_copy   = fast_atts.copy() # so we can delete stuff from fast atts and not mess with iterator
    all_missing_fast = True
    first_loop       = True
    n_missing_denom  = 0

    if fast_data.empty: 
        print(f"!!! no fast data to write for date {date}")
        fast_q.put(False); return False

    for var_name, var_atts in fast_atts_copy.items():
        if var_name == time_name: continue
        perc_miss = fl.perc_missing(fast_data[var_name].values)
        if perc_miss < 100: all_missing_fast = False

    # if there's no data, bale
    if all_missing_fast:
        print("!!! no data on day {}, returning from write without writing".format(date))
        fast_q.put(False); return False

    perc_miss = round((fast_data.index.size/1728000)*100.,2)
    print("... writing level1 fast data on {}, ~{}% of fast data is present".format(date, perc_miss))

    out_dir       = level1_dir+"/"
    file_str_fast = '/fastsled.level1.{}.{}.nc'.format(sled_name, date.strftime('%Y%m%d.%H%M%S'))
    lev1_fast_name  = '{}/{}'.format(out_dir, file_str_fast)
    global_atts_fast = define_global_atts(sled_name,"fast") # global atts for level 1 and level 2
    netcdf_lev1_fast  = Dataset(lev1_fast_name, 'w', zlib=True, clobber=True)

    netcdf_lev1_fast.createDimension('time', None)
    for att_name, att_val in global_atts_fast.items(): # write the global attributes to fast
        netcdf_lev1_fast.setncattr(att_name, att_val)

    try:
        fms = fast_data.index[0]
    except Exception as e:
        print("\n\n\n!!! Something went wrong, there's no fast data today")
        print("... the code doesn't handle that currently")
        slow_q.put(False); return False

    # base_time, ARM spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm =  np.datetime64(today_midnight)

    base_fast = netcdf_lev1_fast.createVariable('base_time', 'i') # seconds since
    base_fast[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])      # seconds

    base_atts = {'string'     : '{}'.format(bot),
                 'long_name' : 'Base time since Epoch',
                 'units'     : 'seconds since {}'.format(et),
                 'ancillary_variables'  : 'time_offset',}
    for att_name, att_val in base_atts.items(): netcdf_lev1_fast['base_time'].setncattr(att_name,att_val)

    # here we create the array and attributes for 'time'
    t_atts_fast   = {'units'     : 'milliseconds since {}'.format(tm),
                     'delta_t'   : '0000-00-00 00:00:00.001',
                     'long_name' : 'Time offset from midnight',
                     'calendar'  : 'standard',}


    bt_atts_fast   = {'units'    : 'milliseconds since {}'.format(bot),
                     'delta_t'   : '0000-00-00 00:00:00.001',
                     'long_name' : 'Time offset from base_time',
                     'calendar'  : 'standard',}

    fast_dti           = pd.DatetimeIndex(fast_data.index.values)
    fast_delta_ints    = np.floor((fast_dti - tm).total_seconds()*1000) # ms 
    t_fast_ind         = pd.Int64Index(fast_delta_ints)
    t_fast             = netcdf_lev1_fast.createVariable('time', 'd','time') # seconds since
    bt_fast_dti        = pd.DatetimeIndex(fast_data.index.values)
    bt_fast_delta_ints = np.floor((bt_fast_dti - bot).total_seconds()*1000) # ms 
    bt_fast_ind        = pd.Int64Index(bt_fast_delta_ints)
    bt_fast            = netcdf_lev1_fast.createVariable('time_offset', 'd','time')

    for att_name, att_val in t_atts_fast.items(): netcdf_lev1_fast['time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts_fast.items(): netcdf_lev1_fast['time_offset'].setncattr(att_name,att_val)

    # this try/except is vestigial, this bug should be fixed
    try:
        t_fast[:] = t_fast_ind.values
        bt_fast[:] = bt_fast_ind.values
    except RuntimeError as re:
        print("!!! there was an error creating fast time variable with netcdf/hd5 I cant debug !!!")
        print("!!! {} !!!".format(re))
        raise re


    for var_name, var_atts in fast_atts.items():
        if var_name == time_name: continue

        var_dtype = fast_data[var_name].dtype
        if var_dtype == object: continue 

        elif fl.column_is_ints(fast_data[var_name]):
        # if issubclass(var_dtype.type, np.integer): # netcdf4 classic doesnt like 64 bit integers
            var_dtype = np.int32
            fill_val  = def_fill_int
            fast_data[var_name].fillna(fill_val, inplace=True)
            var_tmp   = fast_data[var_name].values.astype(np.int32)

        else:
            fill_val  = def_fill_flt
            fast_data[var_name].fillna(fill_val, inplace=True)
            var_tmp   = fast_data[var_name].values
        
        try:
            var_fast = netcdf_lev1_fast.createVariable(var_name, var_dtype, 'time', zlib=True)
            var_fast[:] = var_tmp # compress fast data via zlib=True

        except Exception as e:
            print("!!! something wrong with variable {} on date {} !!!".format(var_name, date))
            print(fast_data[var_name])
            print("!!! {} !!!".format(e))
            continue

        for att_name, att_desc in var_atts.items(): # write atts to the var now
            netcdf_lev1_fast[var_name].setncattr(att_name, att_desc)

        # add a percent_missing attribute to give a first look at "data quality"
        perc_miss = fl.perc_missing(var_fast)
        netcdf_lev1_fast[var_name].setncattr('percent_missing', perc_miss)
        netcdf_lev1_fast[var_name].setncattr('missing_value'  , fill_val)

    print("... done with fast")
    netcdf_lev1_fast.close() 
    fast_q.put(True); return True

def write_level1_slow(slow_data, slow_atts, date, slow_q):

    time_name   = list(slow_atts.keys())[0] # are slow atts and fast atts always going to be the same?
    max_str_len = 30 # these lines are for printing prettilly

    slow_atts_iter   = slow_atts.copy() # so we can delete stuff from slow atts and not mess with iterator
    slow_atts_copy   = slow_atts.copy() # so we can delete stuff from slow atts and not mess with iterator
    all_missing_slow = True 
    first_loop       = True
    n_missing_denom  = 0

    for var_name, var_atts in slow_atts_iter.items():
        if var_name == time_name: continue
        try: dt = slow_data[var_name].dtype
        except KeyError as e: 
            del slow_atts_copy[var_name]
            continue
        perc_miss = fl.perc_missing(slow_data[var_name].values)
        if perc_miss < 100: all_missing_slow = False
        if first_loop: 
            avg_missing_slow = perc_miss
            first_loop=False
        elif perc_miss < 100: 
            avg_missing_slow = avg_missing_slow + perc_miss
            n_missing_denom += 1
    if n_missing_denom > 1: avg_missing_slow = round(avg_missing_slow/n_missing_denom,2)
    else: avg_missing_slow = 100.


    print("... writing level1 for {} on {}, ~{}% of slow data is present".format(sled_name, date, 100-avg_missing_slow))

    out_dir          = level1_dir+"/"
    file_str_slow    = '/slowsled.level1.{}.{}.nc'.format(sled_name, date.strftime('%Y%m%d.%H%M%S'))
    lev1_slow_name   = '{}/{}'.format(out_dir, file_str_slow)
    global_atts_slow = define_global_atts(sled_name, "slow") # global atts for level 1 and level 2
    netcdf_lev1_slow = Dataset(lev1_slow_name, 'w', zlib=True, clobber=True)

    # unlimited dimension to show that time is split over multiple files (makes dealing with data easier)
    netcdf_lev1_slow.createDimension('time', None)
    for att_name, att_val in global_atts_slow.items(): # write the global attributes to slow
        netcdf_lev1_slow.setncattr(att_name, att_val)
        
    try:
        fms = slow_data.index[0]
    except Exception as e:
        print("\n\n\n!!! Something went wrong, there's no slow data today")
        print("... the code doesn't handle that currently")
        slow_q.put(False); return False
        
    # base_time, ARM spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm =  np.datetime64(today_midnight)

    # first write the int base_time, the temporal distance from the UNIX epoch
    base_slow = netcdf_lev1_slow.createVariable('base_time', 'i') # seconds since
    base_slow[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])      # seconds

    base_atts = {'string'     : '{}'.format(bot),
                 'long_name' : 'Base time since Epoch',
                 'units'     : 'seconds since {}'.format(et),
                 'ancillary_variables'  : 'time_offset',}
    for att_name, att_val in base_atts.items(): netcdf_lev1_slow['base_time'].setncattr(att_name,att_val)

    # here we create the array and attributes for 'time'
    t_atts_slow   = {'units'     : 'seconds since {}'.format(tm),
                     'delta_t'   : '0000-00-00 00:01:00',
                     'long_name' : 'Time offset from midnight',
                     'calendar'  : 'standard',}

    bt_atts_slow   = {'units'     : 'seconds since {}'.format(bot),
                     'delta_t'   : '0000-00-00 00:01:00',
                     'long_name' : 'Time offset from base_time',
                     'calendar'  : 'standard',}


    slow_dti = pd.DatetimeIndex(slow_data.index.values)
    slow_delta_ints = np.floor((slow_dti - tm).total_seconds())      # seconds
    t_slow_ind = pd.Int64Index(slow_delta_ints)
    # set the time dimension and variable attributes to what's defined above
    t_slow = netcdf_lev1_slow.createVariable('time', 'd','time') # seconds since

    # now we create the array and attributes for 'time_offset'
    bt_slow_dti = pd.DatetimeIndex(slow_data.index.values)   
    bt_slow_delta_ints = np.floor((bt_slow_dti - bot).total_seconds())      # seconds
    bt_slow_ind = pd.Int64Index(bt_slow_delta_ints)

    # set the time dimension and variable attributes to what's defined above
    bt_slow = netcdf_lev1_slow.createVariable('time_offset', 'd','time') # seconds since

    # this try/except is vestigial, this bug should be fixed
    try:
        t_slow[:] = t_slow_ind.values
        bt_slow[:] = bt_slow_ind.values
    except RuntimeError as re:
        print("!!! there was an error creating slow time variable with netcdf/hd5 I cant debug !!!")
        print("!!! {} !!!".format(re))
        raise re

    for att_name, att_val in t_atts_slow.items(): netcdf_lev1_slow['time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts_slow.items(): netcdf_lev1_slow['time_offset'].setncattr(att_name,att_val)

    for var_name, var_atts in slow_atts_copy.items():
        if var_name == time_name: continue

        var_dtype = slow_data[var_name].dtype
        if fl.column_is_ints(slow_data[var_name]):
        # if issubclass(var_dtype.type, np.integer): # netcdf4 classic doesnt like 64 bit integers
            var_dtype = np.int32
            fill_val  = def_fill_int
            slow_data[var_name].fillna(fill_val, inplace=True)
            var_tmp = slow_data[var_name].values.astype(np.int32)

        else:
            fill_val  = def_fill_flt
            slow_data[var_name].fillna(fill_val, inplace=True)
            var_tmp = slow_data[var_name].values

        var_slow  = netcdf_lev1_slow.createVariable(var_name, var_dtype, 'time', zlib=True)
        var_slow[:]  = var_tmp

        for att_name, att_desc in var_atts.items():
            netcdf_lev1_slow[var_name].setncattr(att_name, att_desc)

        # add a percent_missing attribute to give a first look at "data quality"
        perc_miss = fl.perc_missing(var_slow)
        netcdf_lev1_slow[var_name].setncattr('percent_missing', perc_miss)
        netcdf_lev1_slow[var_name].setncattr('missing_value'  , fill_val)


    netcdf_lev1_slow.close() # close and write files for today
    print("... done with slow")
    slow_q.put(True); return True

# do the stuff to write out the level1 files, set timestep equal to anything from "1min" to "XXmin"
# and we will average the native 1min data to that timestep. right now we are writing 1 and 10min files
def write_turb_netcdf(turb_data, curr_station, date, integration_window, turb_q=None):

    # # temporary debug statements 
    # data_dir = f'./data/ncdf/{curr_station.split("-")[0]}'
    # level1_dir = f'{data_dir}/1_level_ingest_test/'  # where does level1 data go
    # epoch_time = datetime(1970,1,1,0,0,0) # Unix epoch, sets time integers
    # nan = np.NaN
    # def_fill_int = -9999
    # def_fill_flt = -9999.0
    
    if type(turb_data) == type(list()):
        all_data = pd.concat(turb_data)
        turb_data = all_data 

    timestep = str(integration_window)+"min"

    day_delta = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00
    tomorrow  = date+day_delta

    turb_atts, turb_cols = define_turb_variables()

    if turb_data.empty:
        print("... there was no turbulence data to write today {} at station {}".format(date,curr_station))
        try: turb_q.put(False);return False
        except: return False

    # get some useful missing data information for today and print it for the user
    if not turb_data.empty: avg_missing = (1-turb_data.iloc[:,0].notnull().sum()/len(turb_data.iloc[:,1]))*100.
    #fl.perc_missing(turb_data.iloc[:,0].values)
    else: avg_missing = 100.

    print(f"... writing {timestep} turb calcs for {curr_station} on {date.strftime('%F')}, ~{100-avg_missing}% of data is present")
    
    file_str = f"/turbsled.level1.{curr_station}.{date.strftime('%Y%m%d.%H%M%S')}.{timestep}.nc"

    out_dir          = level1_dir+"/"
    turb_name  = '{}/{}'.format(out_dir, file_str)

    global_atts = define_global_atts(curr_station, "turb") # global atts for level 1 and level 2
    netcdf_turb = Dataset(turb_name, 'w', zlib=True, clobber=True)
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

    # base_time, ARM spec, the difference between the time of the first data point and the BOT
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

    if integration_window < 10:
        delt_str = f"0000-00-00 00:0{integration_window}:00"
    else:
        delt_str = f"0000-00-00 00:{integration_window}:00"

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

    # add a percent_missing attribute to give a first loop at "data quality"
    def calc_stats(netcdf_turb, vn, vt): #var_turb
        perc_miss = fl.perc_missing(vt)
        max_val   = np.nanmax(vt) # masked array max/min/etc
        min_val   = np.nanmin(vt)
        avg_val   = np.nanmean(vt)
        netcdf_turb[vn].setncattr('max_val', max_val)
        netcdf_turb[vn].setncattr('min_val', min_val)
        netcdf_turb[vn].setncattr('avg_val', avg_val)
        netcdf_turb[vn].setncattr('percent_missing', perc_miss)
        netcdf_turb[vn].setncattr('missing_value', def_fill_flt)

    # loop over all the data_out variables and save them to the netcdf along with their atts, etc
    for var_name, var_atts in turb_atts.items():

        if turb_data[var_name].isnull().all():
            if turb_data[var_name].dtype == object: # happens when all fast data is missing...
                turb_data[var_name] = np.float64(turb_data[var_name])     

        # create variable, # dtype inferred from data file via pandas
        var_dtype = turb_data[var_name][0].dtype
        if turb_data[var_name][0].size == 1:
            var_turb  = netcdf_turb.createVariable(var_name, var_dtype, 'time')
            calc_stats(netcdf_turb, var_name, turb_data[var_name].values)
            turb_data[var_name].fillna(def_fill_flt, inplace=True)
            # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas data)
            var_turb[:] = turb_data[var_name].values
        else:
            if 'fs' in var_name:  
                netcdf_turb.createDimension('freq', turb_data[var_name][0].size)   
                var_turb  = netcdf_turb.createVariable(var_name, var_dtype, ('freq'))
                #calc_stats(netcdf_turb, var_name, turb_data[var_name].values)

                try: 
                    turb_data[var_name][0].fillna(def_fill_flt, inplace=True)

                # sometimes, in odd cases, this is an ndarray.... so fix that
                except AttributeError as ae:
                    if len(turb_data[var_name][0]) > 1: 
                        turb_data[var_name][0] = pd.Series(turb_data[var_name][0]).fillna(def_fill_flt).values
                    else:
                        raise

                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas
                # data). this is even stupider in multiple dimensions
                try:
                    var_turb[:] = turb_data[var_name][0].values      
                except Exception as ex:
                    if len(turb_data[var_name][0]) > 1: 
                        var_turb[:] = turb_data[var_name][0]
                    else: raise

            else:   
                var_turb  = netcdf_turb.createVariable(var_name, var_dtype, ('time','freq'))
                turb_data[var_name].fillna(def_fill_flt, inplace=True)
                for jj in range(0,turb_data[var_name].size):
                    try: turb_data[var_name][jj].fillna(def_fill_flt, inplace=True)
                    except: do_nothing = True
                # convert DataFrame to np.ndarray and pass data into netcdf (netcdf can't handle pandas
                # data). this is even stupider in multiple dimensions
                tmp = np.empty([turb_data[var_name].size,turb_data[var_name][0].size])
                for jj in range(0,turb_data[var_name].size):
                    try: tmp[jj,:]=np.array(turb_data[var_name][jj])
                    except: i_dont_understand_why_freq_would_change_size_ever = True # probably figure this out...
                var_turb[:] = tmp         

        # add variable descriptions from above to each file
        for att_name, att_desc in var_atts.items(): netcdf_turb[var_name] .setncattr(att_name, att_desc)

    netcdf_turb.close() # close and write files for today

    try: turb_q.put(True);return True
    except: return True


def write_level1_soil(soil_data, soil_atts, date, soil_q):

    time_name   = list(soil_atts.keys())[0] # are soil atts and fast atts always going to be the same?
    max_str_len = 30 # these lines are for printing prettilly

    soil_atts_iter   = soil_atts.copy() # so we can delete stuff from soil atts and not mess with iterator
    soil_atts_copy   = soil_atts.copy() # so we can delete stuff from soil atts and not mess with iterator
    all_missing_soil = True 
    first_loop       = True
    n_missing_denom  = 0

    for var_name, var_atts in soil_atts_iter.items():
        if var_name == time_name: continue
        try: dt = soil_data[var_name].dtype
        except KeyError as e: 
            del soil_atts_copy[var_name]
            continue
        perc_miss = fl.perc_missing(soil_data[var_name].values)
        if perc_miss < 100: all_missing_soil = False
        if first_loop: 
            avg_missing_soil = perc_miss
            first_loop=False
        elif perc_miss < 100: 
            avg_missing_soil = avg_missing_soil + perc_miss
            n_missing_denom += 1
    if n_missing_denom > 1: avg_missing_soil = round(avg_missing_soil/n_missing_denom,2)
    else: avg_missing_soil = 100.


    print("... writing level1 for {} on {}, ~{}% of soil data is present".format(sled_name, date, 100-avg_missing_soil))

    out_dir          = level1_dir+"/"
    file_str_soil    = '/soilsled.level1.{}.{}.nc'.format(sled_name, date.strftime('%Y%m%d.%H%M%S'))
    lev1_soil_name   = '{}/{}'.format(out_dir, file_str_soil)
    global_atts_soil = define_global_atts(sled_name, "soil") # global atts for level 1 and level 2
    netcdf_lev1_soil = Dataset(lev1_soil_name, 'w', zlib=True, clobber=True)

    # unlimited dimension to show that time is split over multiple files (makes dealing with data easier)
    netcdf_lev1_soil.createDimension('time', None)
    for att_name, att_val in global_atts_soil.items(): # write the global attributes to soil
        netcdf_lev1_soil.setncattr(att_name, att_val)
        
    try:
        fms = soil_data.index[0]
    except Exception as e:
        print("\n\n\n!!! Something went wrong, there's no soil data today")
        print("... the code doesn't handle that currently")
        soil_q.put(False); return False
        
    # base_time, ARM spec, the difference between the time of the first data point and the BOT
    today_midnight = datetime(fms.year, fms.month, fms.day)
    beginning_of_time = fms 

    # create the three 'bases' that serve for calculating the time arrays
    et = np.datetime64(epoch_time)
    bot = np.datetime64(beginning_of_time)
    tm =  np.datetime64(today_midnight)

    # first write the int base_time, the temporal distance from the UNIX epoch
    base_soil = netcdf_lev1_soil.createVariable('base_time', 'i') # seconds since
    base_soil[:] = int((pd.DatetimeIndex([bot]) - et).total_seconds().values[0])      # seconds

    base_atts = {'string'     : '{}'.format(bot),
                 'long_name' : 'Base time since Epoch',
                 'units'     : 'seconds since {}'.format(et),
                 'ancillary_variables'  : 'time_offset',}
    for att_name, att_val in base_atts.items(): netcdf_lev1_soil['base_time'].setncattr(att_name,att_val)

    # here we create the array and attributes for 'time'
    t_atts_soil   = {'units'     : 'seconds since {}'.format(tm),
                     'delta_t'   : '0000-00-00 00:01:00',
                     'long_name' : 'Time offset from midnight',
                     'calendar'  : 'standard',}

    bt_atts_soil   = {'units'     : 'seconds since {}'.format(bot),
                     'delta_t'   : '0000-00-00 00:01:00',
                     'long_name' : 'Time offset from base_time',
                     'calendar'  : 'standard',}


    soil_dti = pd.DatetimeIndex(soil_data.index.values)
    soil_delta_ints = np.floor((soil_dti - tm).total_seconds())      # seconds
    t_soil_ind = pd.Int64Index(soil_delta_ints)
    # set the time dimension and variable attributes to what's defined above
    t_soil = netcdf_lev1_soil.createVariable('time', 'd','time') # seconds since

    # now we create the array and attributes for 'time_offset'
    bt_soil_dti = pd.DatetimeIndex(soil_data.index.values)   
    bt_soil_delta_ints = np.floor((bt_soil_dti - bot).total_seconds())      # seconds
    bt_soil_ind = pd.Int64Index(bt_soil_delta_ints)

    # set the time dimension and variable attributes to what's defined above
    bt_soil = netcdf_lev1_soil.createVariable('time_offset', 'd','time') # seconds since

    # this try/except is vestigial, this bug should be fixed
    try:
        t_soil[:] = t_soil_ind.values
        bt_soil[:] = bt_soil_ind.values
    except RuntimeError as re:
        print("!!! there was an error creating soil time variable with netcdf/hd5 I cant debug !!!")
        print("!!! {} !!!".format(re))
        raise re

    for att_name, att_val in t_atts_soil.items(): netcdf_lev1_soil['time'].setncattr(att_name,att_val)
    for att_name, att_val in bt_atts_soil.items(): netcdf_lev1_soil['time_offset'].setncattr(att_name,att_val)

    for var_name, var_atts in soil_atts_copy.items():
        if var_name == time_name: continue

        var_dtype = np.float64
        fill_val  = def_fill_flt
        soil_data[var_name].fillna(fill_val, inplace=True)
        var_tmp = soil_data[var_name].values

        var_soil  = netcdf_lev1_soil.createVariable(var_name, var_dtype, 'time', zlib=True)
        var_soil[:]  = var_tmp

        for att_name, att_desc in var_atts.items():
            netcdf_lev1_soil[var_name].setncattr(att_name, att_desc)

        # add a percent_missing attribute to give a first look at "data quality"
        perc_miss = fl.perc_missing(var_soil)
        netcdf_lev1_soil[var_name].setncattr('percent_missing', perc_miss)
        netcdf_lev1_soil[var_name].setncattr('missing_value'  , fill_val)


    netcdf_lev1_soil.close() # close and write files for today
    print("... done with soil")
    soil_q.put(True); return True


# this runs the function main as the main program... this is a hack that allows functions
# to come after the main code so it presents in a more logical, C-like, way
if __name__ == '__main__':
    main()


