#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
# ############################################################################################
# AUTHOR: Michael Gallagher (CIRES/NOAA)
# EMAIL:  michael.r.gallagher@noaa.gov
# 
# This script takes in the 'slow' level1 data and makes plots of a suite of variables. Each
# plot is given a general name (e.g. meteorology) and then an associated dictionary controls
# the number of subplots and the variables that go on each subplot. 
#
# The data ingest and the plots are done in parallel, so your lap could get a little hot...
# ... but it means the full year of data and make *all* of the daily plots in less than 20
# minutes on an SSD based system. Slower for rusty stuff. You can do this too by running:
#
# /usr/bin/time --format='%C ran in %e seconds' ./plot_asfs_lev1_quicklooks.py -v -s 20191005 -e 20201002 
#
# This scripts requires 4+ threads so if you do this on a dual-core system it's going to be
# a little disappointing.
# 
#
# DATES:
#   v1.0: 2020-11-13
# 
# ############################################################################################
import calendar 

from datetime  import datetime, timedelta
from multiprocessing import Process as P
from multiprocessing import Queue   as Q

import os, inspect, argparse, sys, socket
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorsys

global nthreads 
hostname = socket.gethostname()
if '.psd.' in hostname:
    if hostname.split('.')[0] in ['linux1024', 'linux512']:
        nthreads = 25  # the twins have 32 cores/64 threads, won't hurt if we use <30 threads
    elif hostname.split('.')[0] in ['linux64', 'linux128', 'linux256']:
        nthreads = 12  # 
    else:
        nthreads = 90  # the new compute is hefty.... real hefty

else: nthreads = 8     # laptops don't tend to have 64  cores... yet


# need to debug something? this makes useful pickle files in ./tests/ ... uncomment below if you want to kill threading
we_want_to_debug = False
if we_want_to_debug:

    # from multiprocessing.dummy import Process as P
    # from multiprocessing.dummy import Queue   as Q
    # nthreads = 1
    try: from debug_functions import drop_me as dm
    except: you_dont_care=True

# this is here because for some reason the default matplotlib doesn't
# like running headless...  off with its head
mpl.use('pdf');
mpl.interactive(False)

plt.ioff() # turn off pyplot interactive mode 
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'  # just in case
os.environ['HDF5_MPI_OPT_TYPES']='TRUE'  # just in case

import numpy  as np
import pandas as pd
import xarray as xr

sys.path.insert(0,'../')

import functions_library as fl 
from get_data import get_splash_data

#import warnings
mpl.warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning) 
mpl.warnings.filterwarnings("ignore", category=UserWarning) 
mpl.warnings.filterwarnings("ignore", category=RuntimeWarning) 
mpl.warnings.filterwarnings("ignore", category=FutureWarning) 

def main(): # the main data crunching program

    plot_style = "seaborn-whitegrid"

    default_data_dir = '/PSL/Observations/Campaigns/SPLASH/' # give '-p your_directory' to the script if you don't like this

    make_daily_plots   = True
    make_monthly_plots = True # make plots that include data from each leg

    global sled_dict
    sled_dict = {'asfs50': ' Avery Picnic', 'asfs30': 'Kettle Ponds'} # file names

    # arbitrary date ranges, if you would like....
    make_leg_plots = False
    leg1_start = datetime(2021,10,1)
    leg2_start = datetime(2021,11,1) 
    leg3_start = datetime(2021,12,1)
    leg4_start = datetime(2022,1,1)
    leg5_start = datetime(2022,2,1)
    mosaic_end = datetime(2022,10,2)

    leg_list   = [leg1_start,leg2_start,leg3_start,leg4_start,leg5_start,mosaic_end]
    leg_names = ["Oct","Nov","Dec","Jan","After_Jan"]

    global sleds_to_plot, code_version # code_version is the *production* code and is pulled in from nc files later
    sleds_to_plot = ('asfs30','asfs50')

    # what are we plotting? dict key is the full plot name, value is another dict...
    # ... of subplot titles and the variables to put on each subplot
    var_dict                     = {}
    var_dict['meteorology']      = {'temperature'      : ['apogee_targ_T_Avg','vaisala_T_Avg'],
                                   'humidity'          : ['vaisala_RH_Avg'],
                                   'pressure'          : ['vaisala_P_Avg'],
                                   }
    var_dict['winds']            = {'winds_horizontal' : ['metek_horiz_Avg'], # CREATED BELOW
                                   'winds_vertical'    : ['metek_z_Avg'],
                                   'tilt'              : ['sr30_swd_tilt_Avg', 'sr30_swu_tilt_Avg'],
                                   }
    var_dict['radiation']        = {'shortwave'        : ['sr30_swu_Irr_Avg','sr30_swd_Irr_Avg'], 
                                   'longwave'          : ['ir20_lwu_Wm2_Avg','ir20_lwd_Wm2_Avg'], 
                                   'net'               : ['net_Irr_Avg'], # CREATED BELOW
                                   }
    var_dict['plates_and_sr50']  = {'flux_plates'      : ['fp_A_Wm2_Avg','fp_B_Wm2_Avg'],
                                   'surface_distance'  : ['sr50_dist_Avg'],
                                   }
    var_dict['is_alive']         = {'logger_temp'      : ['PTemp_Avg'],
                                   'logger_voltage'    : ['batt_volt_Avg'],
                                   'gps'               : ['gps_hdop_Avg'],
                                   }
    var_dict['lat_lon']          = {'minutes'          : ['gps_lat_min_Avg', 'gps_lon_min_Avg'],
                                   'degrees'           : ['gps_lat_deg_Avg', 'gps_lon_deg_Avg'],
                                   'heading'           : ['gps_hdg_Avg'],
                                  }
    var_dict['licor']            = {'gas'              : ['licor_co2_Avg','licor_h2o_Avg'],
                                   'quality'           : ['licor_co2_str_out_Avg'],
                                  }

    var_dict['turb_energy']      = {'seb'              : ['Hs','bulk_Hs','Hl', 'bulk_Hl'],
                                     'winds'            : ['metek_horiz_Avg', 'metek_z_Avg'],
                                     'ustar'            : ['ustar'], 
                                     }

    var_dict['turb_eval']         = {'sigw/ustar'       : (['sigW/ustar'], ['wspd_vec_mean']), 
                                     'sigw/ustar '      : (['sigW/ustar'], ['wdir_vec_mean']), 
                                     }

    var_dict['ec_vs_bulk_hs']     = {'EC Hs'       : (['Hs'], ['bulk_Hs']),
                                     }

    var_dict['ec_vs_bulk_ustar']     = {'EC ustar'       : (['ustar'], ['bulk_ustar']),
                                     }


    unit_dict                    = {}
    unit_dict['meteorology']     = {'temperature'      : 'C',
                                    'humidity'         : '%', 
                                    'pressure'         : 'hPa', 
                                    }
    unit_dict['winds']           = {'winds_horizontal' : 'm/s', 
                                    'winds_vertical'   : 'm/s',
                                    'tilt'             : 'degrees', 
                                    }
    unit_dict['radiation']       = {'shortwave'        : 'W/m2', 
                                    'longwave'         : 'W/m2', 
                                    'net'              : 'W/m2', 
                                    }
    unit_dict['plates_and_sr50'] = {'flux_plates'      : 'W/m2', 
                                    'surface_distance' : 'm', 
                                    }
    unit_dict['is_alive']        = {'logger_temp'      : 'C', 
                                    'logger_voltage'   : 'V',
                                    'gps'              : 'none', 
                                    }
    unit_dict['lat_lon']         = {'minutes'          : 'minutes',
                                    'degrees'          : 'degrees',
                                    'heading'          : 'degrees'
                                    }
    unit_dict['licor']           = {'gas'              : 'g/m3',
                                    'quality'          : '%',
                                    }
    unit_dict['turb_energy']     = {'seb'              : 'W/m2',
                                    'winds'            : 'm/s',
                                    'ustar'            : 'm/s',
                                    }

    unit_dict['turb_eval']     = {'sigw/ustar'         : ['', 'm/s'], 
                                  'sigw/ustar '        : ['', '$\deg'],
                                  }

    unit_dict['ec_vs_bulk_hs'] = {'EC Hs' : ['W/m2','W/m2'],
                                  }

    unit_dict['ec_vs_bulk_ustar'] = {'EC ustar' : ['m/s', 'm/s'],    
                                  }

    # ###########################################################

    # if you put a color in the list, (rgb or hex) the function below will all lines different luminosities
    # of the same hue. if you put a 3-tuple of colors, it will use the colors provided explicitly for 30/40/50
    color_dict = {}
    color_dict['meteorology']     = ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8']
    color_dict['winds']           = ['#4878CF','#6ACC65','#C4AD66','#77BEDB','#4878CF']
    color_dict['radiation']       = ['#001C7F','#017517','#8C0900','#7600A1','#B8860B','#006374','#001C7F']
    color_dict['plates_and_sr50'] = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
    color_dict['is_alive']        = [((0.8,0.8,0.9), (0.35,0.55,0.55), (0.55,0.55,0.35)), (0.2,0.5,0.8), (0.1,0.1,0.3)]
    color_dict['lat_lon']         = [(0.8,0.8,0.8),(0.2,0.4,0.2),(0.8,0.6,0.6),
                                     (0.2,0.4,0.2),(0.9,0.5,0.9),(0.3,0.1,0.5),(0.1,0.01,0.01)]
    color_dict['licor']       = ['#23001a','#ffe000','#00fdff']
    color_dict['turb_energy'] = ['#504497', '#cda577','#5f9acd', '#e18395', '#496e66', '#ac4b16','#ffe000','#00fdff']
    color_dict['turb_eval']   = ['#a27398', '#3b8d89']
    color_dict['ec_vs_bulk_hs'] = ['#4bb498', '#c7396f']
    color_dict['ec_vs_bulk_ustar'] = ['#386395', '#723550']
    
    # gg_colors    = ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8']
    # muted_colors = ['#4878CF','#6ACC65','#D65F5F','#B47CC7','#C4AD66','#77BEDB','#4878CF']
    # dark_colors  = ['#001C7F','#017517','#8C0900','#7600A1','#B8860B','#006374','#001C7F']
    # other_dark   = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
    # other_colors = [(0.8,0.8,0.9), (0.85,0.85,0.9), (0.9,0.9,1.0)]

    parser = argparse.ArgumentParser()                                
    parser.add_argument('-v', '--verbose',    action ='count', help='print verbose log messages')            
    parser.add_argument('-s', '--start_time', metavar='str',   help='beginning of processing period, Ymd syntax')
    parser.add_argument('-e', '--end_time',   metavar='str',   help='end  of processing period, Ymd syntax')
    parser.add_argument('-p', '--path', metavar='str', help='base path to data up to, including /data/, include trailing slash')
    parser.add_argument('-pd', '--pickledir', metavar='str',help='want to store a pickle of the data for debugging?')

    args         = parser.parse_args()
    v_print      = print if args.verbose else lambda *a, **k: None
    verboseprint = v_print

    global data_dir, level1_dir # paths
    if args.path: data_dir = args.path
    else: data_dir = default_data_dir
    
    start_time = datetime.today()
    if args.start_time: start_time = datetime.strptime(args.start_time, '%Y%m%d') 
    else: # make the data processing start yesterday! i.e. process only most recent full day of data
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, day=start_time.day-1) 
    if args.end_time: end_time = datetime.strptime(args.end_time, '%Y%m%d')   
    else: end_time = start_time

    if args.pickledir: pickle_dir=args.pickledir
    else: pickle_dir=False

    print('---------------------------------------------------------------------------------------')
    print('Plotting data days between {} -----> {}'.format(start_time,end_time))
    print('---------------------------------------------------------------------------------------\n')

    quicklooks_dir   = '{}/quicklooks_test/1_level/'.format(data_dir)
    out_dir_daily    = '{}/daily/'.format(quicklooks_dir)    # where you want to put the png
    out_dir_all_days = '{}/all_days/'.format(quicklooks_dir) # where you want to put the png

    # plot for all of leg 2.
    day_series = pd.date_range(start_time, end_time) # we're going to get date for these days between start->end

    print(f"Retreiving data from netcdf files... {data_dir}")
    df_list = []
    for station in sleds_to_plot:
        df_station, code_version = get_splash_data(station, start_time, end_time, 1,
                                                   data_dir, 'slow', False, nthreads, pickle_dir)

        # add turbulence variables and interpolate to slow time index
        df_turb, code_version = get_splash_data(station, start_time, end_time, 1,
                                                data_dir, 'turb', False, nthreads, pickle_dir, '10min')
        turb_vars_to_keep = ['Hs', 'Hl', 'bulk_Hl', 'bulk_Hs', 'ustar', 'bulk_ustar']
        #dm({**locals(), **globals()}, 0)
        for v in turb_vars_to_keep: df_station[v] = df_turb[v].reindex(df_turb.index.union(df_station.index)).interpolate('index')
        for v in df_station.columns:
            try: df_turb[v] = df_station[v].reindex(df_station.index.union(df_turb.index)).interpolate('index', limit=10)
            except Exception as e:
                print(e)
                print(f"!!! interp didn't work for {v} !!!")
        df_station = df_station.add_suffix('_{}'.format(station))
        df_turb = df_turb.add_suffix('_{}'.format(station))
        df_list.append(df_turb)
    df = pd.concat(df_list, axis=1)

    ## create variables that we want to have 
    for istation, curr_station in enumerate(sleds_to_plot):
        df['net_Irr_Avg_{}'.format(curr_station)] = \
           df['sr30_swu_Irr_Avg_{}'.format(curr_station)] \
           -df['sr30_swd_Irr_Avg_{}'.format(curr_station)] \
          +df['ir20_lwu_Wm2_Avg_{}'.format(curr_station)] \
          -df['ir20_lwd_Wm2_Avg_{}'.format(curr_station)]

        df['metek_horiz_Avg_{}'.format(curr_station)] = \
          np.sqrt(df['metek_x_Avg_{}'.format(curr_station)]*df['metek_x_Avg_{}'.format(curr_station)]+
                  df['metek_y_Avg_{}'.format(curr_station)]*df['metek_y_Avg_{}'.format(curr_station)])

        df['sigW/ustar_{}'.format(curr_station)] = \
            df['sigW_{}'.format(curr_station)] / df['ustar_{}'.format(curr_station)] 

        df[f'wspd_vec_mean_{curr_station}'] = np.sqrt(df[f'metek_x_Avg_{curr_station}']**2+df[f'metek_x_Avg_{curr_station}']**2)
        df[f'wdir_vec_mean_{curr_station}'] = np.mod((np.arctan2(-1*df[f'metek_x_Avg_{curr_station}'],-1*df[f'metek_y_Avg_{curr_station}']))*180/np.pi,360)

    make_plots_pretty('seaborn-whitegrid') # ... and higher resolution

    if make_daily_plots:
        day_delta  = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00
        nplotthreads = int(np.floor(nthreads/len(var_dict)))
        if (nplotthreads==0): nplotthreads=1
        print(f"~~ making daily plots for all figures with threads {nplotthreads}~~")
        print("----------------------------------------------------")
        for i_day, start_day in enumerate(day_series):
            if i_day % nplotthreads!=0 or i_day>len(day_series): continue
            print("  ... plotting data for day {} (and {} days *with {} plots* after in parallel)".format(start_day, nplotthreads, len(var_dict)))

            plot_q_list = []; plot_p_list = []
            for ithread in range(0,nplotthreads):
                today     = start_day + timedelta(ithread)
                tomorrow  = today+day_delta
                start_str = today.strftime('%Y-%m-%d') # get date string for file name
                end_str   = (today+timedelta(1)).strftime('%Y-%m-%d')   # get date string for file name

                plot_q_list.append({}); plot_p_list.append({});
                for plot_name, subplot_dict in var_dict.items():
                    plot_q_list[ithread][plot_name] = Q()
                    save_str  ='{}/{}/SPLASH_{}_{}_to_{}.png'.format(out_dir_daily, plot_name, plot_name, start_str, end_str)

                    plot_p_list[ithread][plot_name] = \
                      P(target=make_plot,
                        args=(df[today:tomorrow].copy(), subplot_dict, unit_dict[plot_name],
                              color_dict[plot_name], save_str,True,
                              plot_q_list[ithread][plot_name])).start()


            for ithread in range(0, nplotthreads):
                for plot_name, subplot_dict in var_dict.items():
                    plot_q_list[ithread][plot_name].get()

    make_plots_pretty(plot_style)

    if make_monthly_plots: 
        month_list = [day_series[0].replace(day=1)]
        for d in day_series: # get list of months in the date range
            if d.replace(day=1) > month_list[len(month_list)-1]: month_list.append(d.replace(day=1))

        print("   ... making monthly plots for ",end='', flush=True)
        for month in month_list:
            print(" {}...".format(month.month_name()),end='', flush=True)
            leg_dir   = "{}/{}{}".format(quicklooks_dir, month.month_name(), month.year)
            start_day = month; 
            end_day   = month+pd.tseries.offsets.MonthEnd()

            ifig = -1
            plot_q_dict = {}; plot_p_dict = {}
            for plot_name, subplot_dict in var_dict.items():

                ifig += 1; plot_q_dict[plot_name] = Q()
                save_str  ='{}/SPLASH_{}_{}_{}.png'.format(leg_dir, plot_name, month.month_name(), month.year)
                plot_p_dict[plot_name] = P(target=make_plot,
                                           args=(df[start_day:end_day].copy(),subplot_dict,unit_dict[plot_name],
                                                 color_dict[plot_name], save_str, False,plot_q_dict[plot_name])).start()

            for plot_name, subplot_dict in var_dict.items():
                plot_q_dict[plot_name].get()

    make_plots_pretty(plot_style)
    if make_leg_plots:
        print("   ... making leg plots for ",end='', flush=True)
       
        for ileg in range(0,len(leg_list)-1):
            print(" {}...".format(leg_names[ileg]),end='', flush=True)
            leg_dir   = "{}/{}_complete".format(quicklooks_dir,leg_names[ileg])
            start_day = leg_list[ileg]; start_str = start_day.strftime('%Y-%m-%d') # get date string for file name
            end_day   = leg_list[ileg+1]; end_str = end_day.strftime('%Y-%m-%d')   # get date string for file name

            ifig = -1
            plot_q_dict = {}; plot_p_dict = {}
            for plot_name, subplot_dict in var_dict.items():

                ifig += 1; plot_q_dict[plot_name] = Q()
                save_str  ='{}/SPLASH_{}_{}_to_{}.png'.format(leg_dir, plot_name, start_str, end_str)
                plot_p_dict[plot_name] = P(target=make_plot,
                                           args=(df[start_day:end_day].copy(),subplot_dict,unit_dict[plot_name],
                                                 color_dict[plot_name], save_str, False,plot_q_dict[plot_name])).start()

            for plot_name, subplot_dict in var_dict.items():
                plot_q_dict[plot_name].get()

    # make plots for range *actually* requested when calling scripts
    start_str = start_time.strftime('%Y-%m-%d') # get date string for file name
    end_str   = end_time.strftime('%Y-%m-%d')   # get date string for file name

    ifig = -1
    plot_q_dict = {}; plot_p_dict = {}
    for plot_name, subplot_dict in var_dict.items():
        ifig += 1; plot_q_dict[plot_name] = Q()
        save_str  ='{}/FluxStations_{}_{}_to_{}.png'.format(out_dir_all_days, plot_name, start_str, end_str)
        plot_p_dict[plot_name] = P(target=make_plot,
                                   args=(df[start_time:end_time].copy(), subplot_dict, unit_dict[plot_name],
                                         color_dict[plot_name], save_str,False,plot_q_dict[plot_name])).start()

    for plot_name, subplot_dict in var_dict.items():
        plot_q_dict[plot_name].get()

    plt.close('all') # closes figure before looping again 
    exit() # end main()

def get_asfs_data(curr_file, curr_station, today, q):

    if os.path.isfile(curr_file):
        xarr_ds = xr.open_dataset(curr_file)
        data_frame = xarr_ds.to_dataframe()
        data_frame = data_frame.add_suffix('_{}'.format(curr_station))
        code_version = xarr_ds.attrs['version']
    else:
        print(' !!! file {} not found for date {}'.format(curr_file,today))
        data_frame   = pd.DataFrame()
        code_version = None
    q.put(data_frame)
    q.put(code_version)
    return # can be implicit

# abstract plotting to function so plots are made iteratively according to the keys and values in subplot_dict and
# the supplied df and df.index.... i.e. this plots the full length of time available in the supplied df
def make_plot(df, subplot_dict, units, colors, save_str, daily, q):

    nsubs = len(subplot_dict)
    if daily:     fig, ax = plt.subplots(nsubs,1,figsize=(80,40*nsubs))  # square-ish, for daily detail
    elif nsubs>1: fig, ax = plt.subplots(nsubs,1,figsize=(160,30*nsubs)) # more oblong for long time series
    else:         fig, ax = plt.subplots(nsubs,1,figsize=(80,80)) # more oblong for long time series

    # loop over subplot list and plot all variables for each subplot
    isub = -1
    for subplot_name, orig_var_list in subplot_dict.items():

        if type(orig_var_list) == type(tuple()):
            is_scatter = True
            x_var_list = orig_var_list[1]
            var_list   = orig_var_list[0]
            x_units    = units[subplot_name][1]
            units[subplot_name] = units[subplot_name][0]

        else:
            is_scatter = False
            var_list   = orig_var_list 

        isub+=1
        legend_additions = [] # uncomment code below to add the percent of missing data to the legend

        for ivar, var in enumerate(var_list):
            if isinstance(colors[ivar],str) or isinstance(colors[ivar][0],float) :
                color_tuples = get_rgb_trio(colors[ivar])
            else:
                color_tuples = list(colors[ivar])
            color_tuples = normalize_luminosity(color_tuples)

            for istation, curr_station in enumerate(sleds_to_plot):
                asfs_var   = var+'_{}'.format(curr_station)
                if istation==0: 
                    var_min = (df[asfs_var].median()-5*df[asfs_var].std())
                    var_max = (df[asfs_var].median()+5*df[asfs_var].std())
                else:
                    var_min = (var_min+df[asfs_var].median()-5*df[asfs_var].std())/2
                    var_max = (var_max+df[asfs_var].median()+5*df[asfs_var].std())/2

            # manual override here...
            if 'fp_' in var: 
                var_min = -20; var_max = 20
            
            var_lims = (var_min, var_max)
            for istation, curr_station in enumerate(sleds_to_plot):
                try:
                    asfs_var   = var+'_{}'.format(curr_station)
                    asfs_color = color_tuples[istation]
                    perc_miss  = fl.perc_missing(df[asfs_var])

                    time_lims = (df.index[0], df.index[-1]+(df.index[-1]-df.index[-2])) 
                    if type(ax) != type(np.ndarray(1)) and type(ax) != type(list()):
                        #print(f'wuttttttt {type(ax)}')
                        ax = [ax]
                    if is_scatter == False: 
                        df[asfs_var].plot(xlim=time_lims, ylim=var_lims, ax=ax[isub], color=asfs_color)
                    else: 
                        print(f"plotting... {subplot_name} for {curr_station} — {var_lims}")
                        x_var = f'{x_var_list[ivar]}_{curr_station}'
                        x_vals = df[x_var].values
                        ax[isub].scatter(x=x_vals, y=df[asfs_var].values, color=asfs_color, s=900, alpha=0.6, label=curr_station)
                        ax[isub].set_ylim(var_lims)
                        ax[isub].set_xlim((np.nanmin(x_vals), np.nanmax(x_vals)))
                        ax[isub].set_xlabel(f'{x_var_list[ivar]} [{x_units}]', labelpad=100)

                        numavgd = 24; npts = int(len(df)/numavgd)
                        ndf = df[{x_var, asfs_var}]; ndf.index = ndf[x_var].values; ndf = ndf.sort_index()
                        sdf = pd.DataFrame(index=range(npts))

                        sdf[x_var] = [ndf[x_var].iloc[i*numavgd:(i+1)*numavgd].mean() for i in range(int(npts))]
                        sdf[asfs_var] = [ndf[asfs_var].iloc[i*numavgd:(i+1)*numavgd].mean() for i in range(int(npts))]

                        sdf = sdf.rolling(5, min_periods=1).mean()

                        import matplotlib.patheffects as pe
                        ax[isub].plot(sdf[x_var].values, sdf[asfs_var].values, color=asfs_color, linewidth=25,
                                      path_effects = [pe.SimpleLineShadow(alpha=0.8, shadow_color='white',**{'linewidth':50}), pe.Normal()]) 

                    legend_additions.append('{}_{} (missing {}%)'.format(asfs_var, sled_dict[curr_station], str(perc_miss)))
                    plot_success = True
                except Exception as e:
                    print(f"... !!! FAILURE {asfs_var} {subplot_name} !!!\n      -----------> {e} ")
                    import traceback
                    #traceback.print_exc()
                    legend_additions.append('{} (no data)'.format(asfs_var))
                    continue

        if is_scatter:
            y = np.linspace(var_lims[0],var_lims[1],100)
            if x_units==units[subplot_name]: ax[isub].plot(y,y,'r--')#, label='1:1')

        #add useful data info to legend
        try: h,l = ax[isub].get_legend_handles_labels()
        except: print(f"... {subplot_name}... FAILED!!!!!!")
        for s in range(0,len(legend_additions)): l[s] = legend_additions[s]

        #ax[isub].legend(l, loc='upper right',facecolor=(0.3,0.3,0.3,0.5),edgecolor='white')
        ax[isub].legend(l, loc='best',facecolor=(0.3,0.3,0.3,0.5),edgecolor='white')    
        ax[isub].set_ylabel('{} [{}]'.format(subplot_name, units[subplot_name]))
        ax[isub].grid(b=True, which='major', color='grey', linestyle='-')
        #ax[isub].grid(b=False, which='minor')

        if isub==len(subplot_dict)-1 and is_scatter == False:
            ax[isub].set_xlabel('date [UTC]', labelpad=100)
        elif isub>len(subplot_dict)-1 and is_scatter==False:
            ax[isub].tick_params(which='both',labelbottom=False)
            ax[isub].set_xlabel('', labelpad=-200)

    fig.text(0.5, 0.005,'(plotted on {} from level1 data version {} )'.format(datetime.today(), code_version),
             ha='center')

    fig.tight_layout(pad=2.0)
    #fig.tight_layout(pad=5.0) # cut off white-space on edges

    #print('... saving to: {}'.format(save_str))
    if not os.path.isdir(os.path.dirname(save_str)):
        print("\n!!! making directory {}... hope that's what you intended".format(os.path.dirname(save_str)))
        try: os.makedirs(os.path.dirname(save_str))
        except: do_nothing = True # race condition in multi-threading

    fig.savefig(save_str)
        
    plt.close() # closes figure before exiting
    q.put(True)
    return # not necessary

def normalize_luminosity(color_tuples):
    return_colors = []
    pre_lume_list = [colorsys.rgb_to_hls(r,g,b)[1] for r,g,b in color_tuples]
    for r,g,b in color_tuples: 
        h,l,s = colorsys.rgb_to_hls(r,g,b)
        if l == min(pre_lume_list): return_colors.append(colorsys.hls_to_rgb(h, 0.75, s))
        elif l==max(pre_lume_list): return_colors.append(colorsys.hls_to_rgb(h, 0.25, s))
        else:                       return_colors.append(colorsys.hls_to_rgb(h, 0.5, s))

    return return_colors

# returns 3 rgb tuples of varying darkness for a given color, 
def get_rgb_trio(color):
    if isinstance(color, str):
        rgb = hex_to_rgb(color)
    else: rgb = color
    r=rgb[0]; g=rgb[1]; b=rgb[2]
    lume = np.sqrt(0.299*r**22 + 0.587*g**2 + 0.114*b**2)
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    if(lume>0.5): 
        col_one = colorsys.hls_to_rgb(h, l, s)
        col_two = colorsys.hls_to_rgb(h, l-0.2, s)
        col_thr = colorsys.hls_to_rgb(h, l-0.4, s)
    else:
        col_one = colorsys.hls_to_rgb(h, l+0.4, s)
        col_two = colorsys.hls_to_rgb(h, l+0.2, s)
        col_thr = colorsys.hls_to_rgb(h, l, s)
    return [col_one, col_two, col_thr]

def hex_to_rgb(hex_color):
    rgb_tuple = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return tuple(map(lambda x: x/256.0, rgb_tuple))

def make_plots_pretty(style_name):
    # plt.style.use(plot_style)            # grey grid with bolder colors
    # plt.style.use('seaborn-whitegrid') # white grid with softer colors
    plt.style.use(style_name)

    from matplotlib import font_manager as fm
    fpath = "/home/mgallagher/.local/share/fonts/"
    font_files = fm.findSystemFonts(fontpaths=fpath)
    for font_file in font_files: fm.fontManager.addfont(font_file)

    # flist = fm.get_fontconfig_fonts()
    # names = [fm.FontProperties(fname=fname).get_name() for fname in flist]
    # for n in names: print(n)

    mpl.rcParams['font.size']           = 80
    mpl.rcParams['font.family']         = "Lato"
    mpl.rcParams['legend.fontsize']     = 'medium'


    mpl.rcParams['lines.linewidth']     = 12
    mpl.rcParams['axes.labelsize']      = 'xx-large'
    mpl.rcParams['axes.titlesize']      = 'xx-large'
    mpl.rcParams['xtick.labelsize']     = 'xx-large'
    mpl.rcParams['ytick.labelsize']     = 'xx-large'
    mpl.rcParams['ytick.labelsize']     = 'xx-large'
    mpl.rcParams['grid.linewidth']      = 1.
    mpl.rcParams['grid.color']          = 'gainsboro'
    mpl.rcParams['axes.linewidth']      = 6
    mpl.rcParams['axes.grid']           = True
    mpl.rcParams['axes.grid.which']     = 'minor'
    mpl.rcParams['axes.edgecolor']      = 'grey'
    mpl.rcParams['axes.labelpad']       = 100
    mpl.rcParams['axes.titlepad']       = 100
    mpl.rcParams['axes.xmargin']        = 0.3
    mpl.rcParams['axes.ymargin']        = 0.3
    mpl.rcParams['xtick.major.pad']     = 50
    mpl.rcParams['ytick.major.pad']     = 50
    mpl.rcParams['xtick.minor.pad']     = 50
    mpl.rcParams['ytick.minor.pad']     = 50
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['axes.spines.right']   = False
    mpl.rcParams['axes.spines.top']     = False
    mpl.rcParams['legend.facecolor']    = 'white'

# this runs the function main as the main program... this is a hack that allows functions
# to come after the main code so it presents in a more logical, C-like, way 
if __name__ == '__main__': 
    main() 

