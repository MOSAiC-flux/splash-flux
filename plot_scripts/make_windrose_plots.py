#!/usr/bin/env python3
# #################################################################
# Hey Sara/Bee!! 
#

from datetime import datetime, timedelta # python library for working with dates
import numpy  as np                      # numpy, all sorts of great math stuff
import pandas as pd                      # pandas, a widely used general purpose data processing library
import xarray as xr                      # xarray, another general purpose data library, we use to open files
import os, glob, socket, sys, argparse   # misc useful python libs/functions

import plotly.express as px

import multiprocessing as mp

from debug_functions import drop_me as dm 

if '.psd.' in socket.gethostname():
    nthreads = 60 # the twins have 64 cores, it won't hurt if we use ~30
else: nthreads = 8 # if nthreads < nplots then plotting will not be threaded

sys.path.insert(0,'../')
from get_data import get_splash_data

import warnings
warnings.filterwarnings('ignore') # weird minor warning in pandas to ignore

def main(): # the main data crunching program

    default_data_dir = '/PSL/Observations/Campaigns/SPLASH/' # give '-p your_directory' to the script if you don't like this

    out_dir   = './plots/'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    start_time = datetime(2022, 4, 1)
    end_time   = datetime(2022, 5, 31)
    
    station_list  = ['asfs30', 'asfs50']
    #station_list  = ['ASFS50']

    print("-------------------------------------------------------------------")
    print(f"Getting data files for stations {station_list} between dates:")
    print(f"   {start_time} ---------> {end_time}\n")

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
    start_time = datetime(2021,10,10)
    if args.start_time: start_time = datetime.strptime(args.start_time, '%Y%m%d') 
    else: # make the data processing start yesterday! i.e. process only most recent full day of data
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0, day=start_time.day-1) 
    if args.end_time: end_time = datetime.strptime(args.end_time, '%Y%m%d')   
    else:
        end_time = start_time
        end_time = datetime.today()
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0, day=end_time.day+1) 

    if args.pickledir: pickle_dir=args.pickledir
    else: pickle_dir='./'

    print('---------------------------------------------------------------------------------------')
    print('Plotting data days between {} -----> {}'.format(start_time,end_time))
    print('---------------------------------------------------------------------------------------\n')

    quicklooks_dir   = '{}/quicklooks_test/1_level_test/'.format(data_dir)
    out_dir_daily    = '{}/daily/windrose/'.format(quicklooks_dir)    # where you want to put the png
    out_dir_all_days = '{}/all_days/'.format(quicklooks_dir) # where you want to put the png

    day_series = pd.date_range(start_time, end_time) # we're going to get date for these days between start->end
    day_delta  = pd.to_timedelta(86399999999,unit='us') # we want to go up to but not including 00:00

    # loop over stations and output ascii file for each one
    if not os.path.isdir(os.path.dirname(out_dir_daily)):
        print("\n!!! making directory {}... hope that's what you intended".format(os.path.dirname(out_dir_daily)))
        try: os.makedirs(os.path.dirname(save_str))
        except: do_nothing = True # race condition in multi-threading

    for station in station_list: 

        df_station, code_version = get_splash_data(station, start_time, end_time, 1,
                                                   data_dir, 'slow', False, nthreads, pickle_dir)

        arg_list = [ ]

        print(df_station)
        for day_start in pd.date_range(start_time, end_time, freq='1D'):
            day_end = day_start+day_delta
            day_data = df_station[day_start:day_end]
            arg_list.append([df_station, station, out_dir_daily])

        with mp.Pool(processes=nthreads) as pool:                                                                                                               pool.map(make_windrose_threaded, arg_list)
                # import traceback
                # import sys
                # print(traceback.format_exc())

    print("\n\n-------------------------------------------------------------------")
    print("All done!!!")
    print("-------------------------------------------------------------------")

# wrapper that accepts list argument
def make_windrose_threaded(arg_list): make_windrose(arg_list[0], arg_list[1], arg_list[2])

def make_windrose(df, station_name, plot_dir):

    ndf, pdf = bin_windspeeds(df)
    if df.index[0].day == 1: print(f"... plotting {df.index[0].strftime('%m %d %Y')} for {station_name}")

    # Available templates:
    #     ['ggplot2', 'seaborn', 'simple_white', 'plotly',
    #      'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
    #      'ygridoff', 'gridon', 'none']

    fig = px.bar_polar(pdf, r="frequency", theta="direction", color="speed (m/s)",
                       template="plotly_white", color_discrete_sequence=px.colors.sequential.Plasma_r,
                       title=f"{station_name}")

    fig.update_layout(margin=dict(l=0, r=00, t=50, b=50),#paper_bgcolor="LightSteelBlue",
                      legend=dict(x=.85, y=1.1, traceorder="normal",
                                  font=dict( family="sans-serif", size=12, color="black" ),)
                      )

    plot_date = df.index[0]
    fig.add_annotation(text=f'windrose histogram for {plot_date.month}-{plot_date.day}-{plot_date.year} from '+
                       f'{plot_date.hour}:00 to {(plot_date+timedelta(hours=1)).hour}:00', align="center", y=-0.18)

    subdir = plot_dir+f'./{plot_date.month}_{plot_date.day}_{plot_date.year}/'
    if not os.path.exists(subdir): os.makedirs(subdir)
    fig.write_image(f'{subdir}/{plot_date.hour}h_{station_name}_Windrose_'+
                    f'{plot_date.month}_{plot_date.day}_{plot_date.year}.png', scale=5)

# this function calculates the frequency for each wind direction and the strength
# to match the data format expected by plotly... takes some effort
# it returns a new dataframe with the number of counts for each windspeed and direction (*not* a timeseries)
def bin_windspeeds(df):

    speed_step = 1; speed_end = 5

    # book-keeping for how we want to bin and label windspeeds 
    speed_bins_edge  = [s for s in range(0, speed_end+1, speed_step)]
    speed_bins_str   = [f'{s}-{s+speed_step}' for s in range(0, speed_end, speed_step)] + [f'{speed_end}+']
    speed_bins_tuple = [f'({s}, {s+speed_step}]' for s in range(0, speed_end, speed_step)] +['nan']
    
    # book-keeping for how we want to binn and label win directions
    dir_bins_edge  = [d-11.25 for d in np.arange(0, 360, 22.5)]
    dir_bins_str   = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW','WSW', 'W', 'WNW', 'NW', 'NNW']
    dir_bins_tuple = [f'({d-11.25}, {d+11.25}]' for d in np.arange(0, 360, 22.5)]

    wdirs   = df['WDir(deg)'].copy()               # make a copy so we can modify it 
    wdirs   = wdirs.where(wdirs<348.75, wdirs-360) # and then fix the discontinuity

    # "cut" (aka bin via pandas) based on bin edges
    speed_cut = pd.cut(df['WSpd(m/s)'], bins=speed_bins_edge).rename("speed (m/s)").astype(str)
    dir_cut   = pd.cut(wdirs, bins=dir_bins_edge).rename("direction").astype(str)

    speed_cut_copy = speed_cut.copy().rename("speed_old")
    dir_cut_copy   = dir_cut.copy().rename("direction_old")

    # relabel the bins to match the strings expected by plotly
    for idir, dir_tuple in enumerate(dir_bins_tuple):       dir_cut    = dir_cut.where(~(dir_cut==dir_tuple), dir_bins_str[idir])
    for ispeed, speed_tuple in enumerate(speed_bins_tuple): speed_cut = speed_cut.where(~(speed_cut==speed_tuple), speed_bins_str[ispeed])

    # put all the data in one frame so we can look at it to verify, if we want
    sd_df = pd.concat([speed_cut, speed_cut_copy, df['WSpd(m/s)'], dir_cut, dir_cut_copy, df['WDir(deg)']], axis=1)

    tot_bins = len(dir_bins_str)*len(speed_bins_str)
    plotly_df = pd.DataFrame(index=range(0,tot_bins))

    # now we are going to do manual binning, because wind roses are plotted as a pie chart based on two dimensions
    # binned by counts in a wind speed and direction category, so do that logic here and put it into a dataframe
    freq_list  = []; dir_list  = []; speed_list = []  # 
    for spdstr in speed_bins_str:
        for dirstr in dir_bins_str: 
            dir_list.append(dirstr) 
            speed_list.append(spdstr) 
            freq_list.append( 100*(len(sd_df[sd_df["direction"]==dirstr][sd_df['speed (m/s)']==spdstr]) /len(df)) )

    plotly_df['frequency']   = freq_list
    plotly_df['direction']   = dir_list
    plotly_df['speed (m/s)'] = speed_list

    # return the labeled original time series, and the plotly histogram categories
    return sd_df, plotly_df

# this runs the function main() as the main program... 
if __name__ == '__main__':
    main()

