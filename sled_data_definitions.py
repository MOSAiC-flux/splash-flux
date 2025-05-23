# ############################################################################################
#
# This file does the heady work of defining the attributes of the parameters observed at the
# flux stations to variable names that make it into the netcdf files.  This file is where you
# can change anything that doesn't look right in the netcdf files. Caveat!: the variable
# names used here are also used in the "create" code that actually builds the netcdf file. If
# you want to change a variable name, please run the "change_code_var_name" script and it will
# find/replace all instances in both pieces of code, here *and* in the "create" code
#
# But! Changing attributes shouldn't affect the create code, so hack away on that. Please use
# only single quotes for variable names and related attributes.
#
# ############################################################################################

import time, os
import numpy as np

# some of the 'create' code assumes that the first entry in the
# variable dictionaries is the time dimension. it's special for stupid
# pandas reasons ... you can't refer to a dataframe index by its name
# ... so order matters.
from collections import OrderedDict

# Heights definition
global boom_location, mast_location, metek_height, licor_height, boomTe_height, boomPr_height, boomRh_height, boom_height
boom_location = 'ASFS boom (top of station)'
mast_location = 'ASFS mast'
metek_height  = '4.62 m'
licor_height  = '4.4 m'
boomTe_height = '2.89 m'
boomPr_height = '2.68 m'
boomRh_height = '2.60 m'   
boom_height   = '2.8 m'  

def code_version():
    cv = ('1.0', '9/18/2023', 'ccox')
    return cv

# file_type must be "slow", "10hz", or "turb"
def define_global_atts(station_name, file_type):
    cv = code_version()
    # global attributes to be written into the netcdf output file
    global_atts = {
        'date_created'     :'{}'.format(time.ctime(time.time())),
        'title'            :'Study of Precipitation, the Lower Atmosphere and Surface for Hydrology (SPLASH) Atmospheric Surface Flux Station (ASFS) data product',
        'institution'      :'NOAA Physical Sciences Laboratory (PSL) and CIRES/University of Colorado',
        'file_creator'     :'Michael R. Gallagher; Christopher J. Cox',
        'creator_email'    :'michael.r.gallagher@noaa.gov; christopher.j.cox@noaa.gov', 
        'Funding'          :'NOAA Physical Sciences Laboratory (PSL)',
        'source'           :'Observations made by flux stations near Gothic, Colorado', 
        'system'           :'{}'.format(station_name),
        'references'       :'', 
        'keywords'         :'Mountain Hydrology, Snow, Supersite, Observations, Flux, Atmosphere',
        'conventions'      :'cf convention variable naming as attribute whenever possible',  
        'history'          :'processed data based on level 1 ingest files',
        'version'          : cv[0]+', '+cv[1], 
    }

    if file_type == "slow":
        global_atts['quality_control']  = 'This Level 1 product is for archival purposes and has undergone minimal data processing and quality control, please contact the authors/PI if you would like to know more.',

    elif file_type == "10hz":
        global_atts['quality_control']  = 'This 10Hz 3-d winds (u,v,w), acoustic temperature, and gas concentration. Minor quality control is in place, including rotation to earth coordinates but the data remains untouched in processing terms.',

    elif file_type == "turb":  # some specifics for the tubulence file
        global_atts['quality_control']  = 'The source data measured at 20 Hz was quality controlled. Variables relevant for quality control of the derived quantities in this file are also supplied, but the derived quantities themselves are NOT quality-controlled.',
        global_atts['methods']          = 'Code developed from routines used by NOAA ETL/PSD3. Original code read_sonic_hr was written by Chris Fairall and later adopted by Andrey Grachev as read_sonic_10Hz_1hr_Tiksi_2012_9m_v2.m, read_sonic_hr_10.m, read_Eureka_sonic_0_hr_2009_egu.m, read_sonic_20Hz_05hr_Materhorn2012_es2',
        global_atts['references']       = 'Grachev et al. (2013), BLM, 147(1), 51-82, doi 10.1007/s10546-012-9771-0; Grachev et al. (2008) Acta Geophysica. 56(1): 142-166; J.C. Kaimal & J.J. Finnigan "Atmospheric Boundary Layer Flows" (1994)','Cox et al. (2023) Scientific Data doi:10.1038/s41597-023-02415-5'
        global_atts['acknowledgements'] = 'Dr. Andrey Grachev (CIRES), Dr. Chris Fairall (NOAA), Dr. Ludovic Bariteau (CIRES)'

    elif file_type == "level2" or file_type == "seb" or file_type == "level3" or file_type == 'seb3':
        global_atts['quality_control']  = 'Quality control in place for the observations used in the derived products. Some suspect data retained with flags. Please use the qc flags.'
        global_atts['qc_flags'] = '-1 = No Data: Instrument was not functional and no data exists.'+os.linesep+\
            '0 = Good: High certainty that data is accurate to within the expected measurement uncertainty.'+os.linesep+\
            '1 = Caution: Use data with caution as there is reason to believe that the data might have a higher uncertainty than expected and/or is adversely impacted in some way.'+os.linesep+\
            '2 = Bad: Data is determined to be clearly erroneous (out of range, does not pass quality control, is adversely impacted in some way, etc). Data has been removed.'

    return OrderedDict(global_atts)


# we are reading SD card data, which was usually saved without headers, but the headers occasionally change when
# logger programs are modified. it is what it is...  the good news is that there are only a dozen or so
# permutations, so we can just store them here
def get_level1_col_headers(ncol,cver):

    # from MGallagher on 2/1/2022
    col_len_97_0 = ["TIMESTAMP", "gps_lat_deg_Avg","gps_lat_min_Avg","gps_lon_deg_Avg","gps_lon_min_Avg","gps_hdg_Avg","gps_alt_Avg","gps_qc","gps_hdop_Avg","gps_nsat_Avg","metek_InclX_Avg","metek_InclY_Avg","PTemp_Avg","batt_volt_Avg","counts_main_Tot","call_time_mainscan_Max","call_time_modbus_sr301_Max","call_time_modbus_sr302_Max","call_time_modbus_vaisala_Max","call_time_sdi1_Max","call_time_sdi2_Max","call_time_efoy_Max","sr30_swu_DegC_Avg","sr30_swu_DegC_Std","sr30_swu_Irr_Avg","sr30_swu_Irr_Std","sr30_swu_IrrC_Avg","sr30_swu_IrrC_Std","sr30_swd_DegC_Avg","sr30_swd_DegC_Std","sr30_swd_Irr_Avg","sr30_swd_Irr_Std","sr30_swd_IrrC_Avg","sr30_swd_IrrC_Std","apogee_body_T_Avg","apogee_body_T_Std","apogee_targ_T_Avg","apogee_targ_T_Std","sr50_dist_Avg","sr50_dist_Std","sr50_qc_Avg","vaisala_RH_Avg","vaisala_RH_Std","vaisala_T_Avg","vaisala_T_Std","vaisala_Td_Avg","vaisala_Td_Std","vaisala_P_Avg","vaisala_P_Std","metek_x_Avg","metek_x_Std","metek_y_Avg","metek_y_Std","metek_z_Avg","metek_z_Std","ir20_lwu_mV_Avg","ir20_lwu_mV_Std","ir20_lwu_Case_R_Avg","ir20_lwu_Case_R_Std","ir20_lwu_DegC_Avg","ir20_lwu_DegC_Std","ir20_lwu_Wm2_Avg","ir20_lwu_Wm2_Std","ir20_lwd_mV_Avg","ir20_lwd_mV_Std","ir20_lwd_Case_R_Avg","ir20_lwd_Case_R_Std","ir20_lwd_DegC_Avg","ir20_lwd_DegC_Std","ir20_lwd_Wm2_Avg","ir20_lwd_Wm2_Std","fp_A_mV_Avg","fp_A_mV_Std","fp_A_Wm2_Avg","fp_A_Wm2_Std","fp_B_mV_Avg","fp_B_mV_Std","fp_B_Wm2_Avg","fp_B_Wm2_Std","licor_co2_Avg","licor_co2_Std","licor_h2o_Avg","licor_h2o_Std","licor_T_Avg","licor_T_Std","licor_co2_str_out_Avg","licor_co2_str_out_Std","sr30_swu_fantach_Avg","sr30_swu_heatA_Avg","sr30_swd_fantach_Avg","sr30_swd_heatA_Avg","ir20_lwu_fan_Avg","ir20_lwd_fan_Avg","efoy_Ubat_Avg","efoy_Laus_Avg","sr30_swu_tilt_Avg","sr30_swd_tilt_Avg"]

    col_out = eval('col_len_'+str(ncol)+'_'+str(cver)) # yup i did that

    return col_out

# we are reading SD card data, which was usually saved without headers, but the headers occasionally change when
# logger programs are modified. it is what it is...  the good news is that there are only a dozen or so
# permutations, so we can just store them here
def get_level1_soil_col_headers(ncol,cver):

    col_len_25_0 = ["TIMESTAMP", "VWC_5cm", "Ka_5cm", "T_5cm", "BulkEC_5cm", "VWC_10cm", "Ka_10cm", "T_10cm", "BulkEC_10cm", "VWC_20cm", "Ka_20cm", "T_20cm", "BulkEC_20cm", "VWC_30cm", "Ka_30cm", "T_30cm", "BulkEC_30cm", "VWC_40cm", "Ka_40cm", "T_40cm", "BulkEC_40cm", "VWC_50cm", "Ka_50cm", "T_50cm", "BulkEC_50cm"]

    col_out = eval('col_len_'+str(ncol)+'_'+str(cver)) # yup i did that

    return col_out

# defines the column names of the logger data for the flux stations and thus also the output variable names in
# the level1 data files that are supposed to be "nearly-raw netcdf records". this returns two things. 1) a list of dictionaries
# where the key is the variable name and dictionary is a list of netcdf attributes. 2) a list of variable names
def define_level1_slow():
    
    lev1_slow_atts = OrderedDict()

    # units defined here, other properties defined in 'update' call below
    lev1_slow_atts['TIMESTAMP']                    = {'units' : 'TS'}
    lev1_slow_atts['gps_lat_deg_Avg']              = {'units' : 'deg'}
    lev1_slow_atts['gps_lat_min_Avg']              = {'units' : 'min'}
    lev1_slow_atts['gps_lon_deg_Avg']              = {'units' : 'deg'}
    lev1_slow_atts['gps_lon_min_Avg']              = {'units' : 'min'}
    lev1_slow_atts['gps_hdg_Avg']                  = {'units' : 'deg'}
    lev1_slow_atts['gps_alt_Avg']                  = {'units' : 'm'}
    lev1_slow_atts['gps_qc']                       = {'units' : 'N'}
    lev1_slow_atts['gps_hdop_Avg']                 = {'units' : 'unitless'}
    lev1_slow_atts['gps_nsat_Avg']                 = {'units' : 'N'}
    lev1_slow_atts['metek_InclX_Avg']              = {'units' : 'deg'}
    lev1_slow_atts['metek_InclY_Avg']              = {'units' : 'deg'}
    lev1_slow_atts['PTemp_Avg']                    = {'units' : 'degC'}
    lev1_slow_atts['batt_volt_Avg']                = {'units' : 'V'}
    lev1_slow_atts['counts_main_Tot']              = {'units' : 'N'}
    lev1_slow_atts['call_time_mainscan_Max']       = {'units' : 'mSec'}
    lev1_slow_atts['call_time_modbus_sr301_Max']   = {'units' : 'mSec'}
    lev1_slow_atts['call_time_modbus_sr302_Max']   = {'units' : 'mSec'}
    lev1_slow_atts['call_time_modbus_vaisala_Max'] = {'units' : 'mSec'}
    lev1_slow_atts['call_time_sdi1_Max']           = {'units' : 'mSec'}
    lev1_slow_atts['call_time_sdi2_Max']           = {'units' : 'mSec'}
    lev1_slow_atts['call_time_efoy_Max']           = {'units' : 'mSec'}
    lev1_slow_atts['sr30_swu_DegC_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['sr30_swu_DegC_Std']            = {'units' : 'degC'}
    lev1_slow_atts['sr30_swu_Irr_Avg']             = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swu_Irr_Std']             = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swu_IrrC_Avg']            = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swu_IrrC_Std']            = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swd_DegC_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['sr30_swd_DegC_Std']            = {'units' : 'degC'}
    lev1_slow_atts['sr30_swd_Irr_Avg']             = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swd_Irr_Std']             = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swd_IrrC_Avg']            = {'units' : 'W/m2'}
    lev1_slow_atts['sr30_swd_IrrC_Std']            = {'units' : 'W/m2'}
    lev1_slow_atts['apogee_body_T_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['apogee_body_T_Std']            = {'units' : 'degC'}
    lev1_slow_atts['apogee_targ_T_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['apogee_targ_T_Std']            = {'units' : 'degC'}
    lev1_slow_atts['sr50_dist_Avg']                = {'units' : 'm'}
    lev1_slow_atts['sr50_dist_Std']                = {'units' : 'm'}
    lev1_slow_atts['sr50_qc_Avg']                  = {'units' : 'unitless'}
    lev1_slow_atts['vaisala_RH_Avg']               = {'units' : '%'}
    lev1_slow_atts['vaisala_RH_Std']               = {'units' : '%'}
    lev1_slow_atts['vaisala_T_Avg']                = {'units' : 'degC'}
    lev1_slow_atts['vaisala_T_Std']                = {'units' : 'degC'}
    lev1_slow_atts['vaisala_Td_Avg']               = {'units' : 'degC'}
    lev1_slow_atts['vaisala_Td_Std']               = {'units' : 'degC'}
    lev1_slow_atts['vaisala_P_Avg']                = {'units' : 'hPa'}
    lev1_slow_atts['vaisala_P_Std']                = {'units' : 'hPa'}
    lev1_slow_atts['metek_x_Avg']                  = {'units' : 'm/s'}
    lev1_slow_atts['metek_x_Std']                  = {'units' : 'm/s'}
    lev1_slow_atts['metek_y_Avg']                  = {'units' : 'm/s'}
    lev1_slow_atts['metek_y_Std']                  = {'units' : 'm/s'}
    lev1_slow_atts['metek_z_Avg']                  = {'units' : 'm/s'}
    lev1_slow_atts['metek_z_Std']                  = {'units' : 'm/s'}
    lev1_slow_atts['ir20_lwu_mV_Avg']              = {'units' : 'mV'}
    lev1_slow_atts['ir20_lwu_mV_Std']              = {'units' : 'mV'}
    lev1_slow_atts['ir20_lwu_Case_R_Avg']          = {'units' : 'ohms'}
    lev1_slow_atts['ir20_lwu_Case_R_Std']          = {'units' : 'ohms'}
    lev1_slow_atts['ir20_lwu_DegC_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['ir20_lwu_DegC_Std']            = {'units' : 'degC'}
    lev1_slow_atts['ir20_lwu_Wm2_Avg']             = {'units' : 'W/m2'}
    lev1_slow_atts['ir20_lwu_Wm2_Std']             = {'units' : 'W/m2'}
    lev1_slow_atts['ir20_lwd_mV_Avg']              = {'units' : 'mV'}
    lev1_slow_atts['ir20_lwd_mV_Std']              = {'units' : 'mV'}
    lev1_slow_atts['ir20_lwd_Case_R_Avg']          = {'units' : 'ohms'}
    lev1_slow_atts['ir20_lwd_Case_R_Std']          = {'units' : 'ohms'}
    lev1_slow_atts['ir20_lwd_DegC_Avg']            = {'units' : 'degC'}
    lev1_slow_atts['ir20_lwd_DegC_Std']            = {'units' : 'degC'}
    lev1_slow_atts['ir20_lwd_Wm2_Avg']             = {'units' : 'W/m2'}
    lev1_slow_atts['ir20_lwd_Wm2_Std']             = {'units' : 'W/m2'}
    lev1_slow_atts['fp_A_mV_Avg']                  = {'units' : 'mV'}
    lev1_slow_atts['fp_A_mV_Std']                  = {'units' : 'mV'}
    lev1_slow_atts['fp_A_Wm2_Avg']                 = {'units' : 'W/m2'}
    lev1_slow_atts['fp_A_Wm2_Std']                 = {'units' : 'W/m2'}
    lev1_slow_atts['fp_B_mV_Avg']                  = {'units' : 'mV'}
    lev1_slow_atts['fp_B_mV_Std']                  = {'units' : 'mV'}
    lev1_slow_atts['fp_B_Wm2_Avg']                 = {'units' : 'W/m2'}
    lev1_slow_atts['fp_B_Wm2_Std']                 = {'units' : 'W/m2'}
    lev1_slow_atts['licor_co2_Avg']                = {'units' : 'mg/m3'}
    lev1_slow_atts['licor_co2_Std']                = {'units' : 'mg/m3'}
    lev1_slow_atts['licor_T_Avg']                  = {'units' : 'deg C'}
    lev1_slow_atts['licor_T_Std']                  = {'units' : 'deg C'}
    lev1_slow_atts['licor_h2o_Avg']                = {'units' : 'g/m3'}
    lev1_slow_atts['licor_h2o_Std']                = {'units' : 'g/m3'}
    lev1_slow_atts['licor_co2_str_out_Avg']        = {'units' : '%'}
    lev1_slow_atts['licor_co2_str_out_Std']        = {'units' : '%'}
    lev1_slow_atts['sr30_swu_fantach_Avg']         = {'units' : 'Hz'}
    lev1_slow_atts['sr30_swu_heatA_Avg']           = {'units' : 'mA'}
    lev1_slow_atts['sr30_swd_fantach_Avg']         = {'units' : 'Hz'}
    lev1_slow_atts['sr30_swd_heatA_Avg']           = {'units' : 'mA'}
    lev1_slow_atts['ir20_lwu_fan_Avg']             = {'units' : 'mV'}
    lev1_slow_atts['ir20_lwd_fan_Avg']             = {'units' : 'mV'}
    lev1_slow_atts['efoy_Error_Max']               = {'units' : 'N'}
    lev1_slow_atts['efoy_FuellSt_Avg']             = {'units' : '%'}
    lev1_slow_atts['efoy_Ubat_Avg']                = {'units' : 'V'}
    lev1_slow_atts['efoy_Laus_Avg']                = {'units' : 'A'}
    lev1_slow_atts['efoy_Tst_Avg']                 = {'units' : 'C'}
    lev1_slow_atts['efoy_Tint_Avg']                = {'units' : 'C'}
    lev1_slow_atts['efoy_Twt_Avg']                 = {'units' : 'C'}
    lev1_slow_atts['sr30_swu_tilt_Avg']            = {'units' : 'deg'}
    lev1_slow_atts['sr30_swd_tilt_Avg']            = {'units' : 'deg'}

    # these are useful, un-qced variables that are produced in the turbulence calculations
    lev1_slow_atts['wspd_u_mean']             = {'units' : 'm/s'}
    lev1_slow_atts['wspd_v_mean']             = {'units' : 'm/s'}
    lev1_slow_atts['wspd_w_mean']             = {'units' : 'm/s'}
    lev1_slow_atts['wspd_vec_mean']           = {'units' : 'm/s'}
    lev1_slow_atts['wdir_vec_mean']           = {'units' : 'degrees'}
    lev1_slow_atts['wspd_u_std']              = {'units' : 'm/s'}
    lev1_slow_atts['wspd_v_std']              = {'units' : 'm/s'}
    lev1_slow_atts['wspd_w_std']              = {'units' : 'm/s'}

        
    lev1_slow_atts['wspd_u_mean']           .update({'long_name': 'Metek u-component',
                                                'cf_name'       : 'eastward_wind',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'v defined positive west-to-east',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
        
    lev1_slow_atts['wspd_v_mean']           .update({'long_name': 'Metek v-component',
                                                'cf_name'       : 'northward_wind',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'v defined positive south-to-north',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
                                                
    lev1_slow_atts['wspd_w_mean']           .update({'long_name': 'Metek w-component',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'w defined positive up',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})

    lev1_slow_atts['wspd_vec_mean']         .update({'long_name': 'average metek wind speed',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
        
    lev1_slow_atts['wdir_vec_mean']         .update({'long_name': 'average metek wind direction',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
    
        
    lev1_slow_atts['wspd_u_std']           .update({ 'long_name': 'u metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})    
    
    lev1_slow_atts['wspd_v_std']           .update({ 'long_name': 'v metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
    
    lev1_slow_atts['wspd_w_std']           .update({ 'long_name': 'w metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})   

    lev1_slow_atts['TIMESTAMP']                    .update({'long_name'     : 'time of measurement',
                                                            'instrument'    : 'CR1000X',
                                                            'methods'       : 'synched to GPS',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['gps_lat_deg_Avg']              .update({'long_name'     : 'latitude degrees from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPRMC, GPGGA, GPGZDA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_lat_min_Avg']              .update({'long_name'     : 'latitide minutes from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPRMC, GPGGA, GPGZDA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_lon_deg_Avg']              .update({'long_name'     : 'longitude degrees from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPRMC, GPGGA, GPGZDA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_lon_min_Avg']              .update({'long_name'     : 'longitude minutes from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPRMC, GPGGA, GPGZDA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_hdg_Avg']                  .update({'long_name'     : 'heading from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'HEHDT',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_alt_Avg']                  .update({'long_name'     : 'altitude from gps at station',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPRMC, GPGGA, GPGZDA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_qc']                       .update({'long_name'     : 'gps fix quality variable',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPGGA; fix quality: 0 = invalid; 1 = gps fix (sps); 2 = dgps fix; 3 = pps fix; 4 = real time kinematic; 5 = float rtk; 6 = estimated (deck reckoning); 7 = manual input mode; 8 = simulation mode',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_hdop_Avg']                 .update({'long_name'     : 'gps Horizontal Dilution Of Precision (HDOP)',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPGGA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['gps_nsat_Avg']                 .update({'long_name'     : 'gps number of tracked satellites',
                                                            'instrument'    : 'Hemisphere V102',
                                                            'methods'       : 'GPGGA',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['metek_InclX_Avg']              .update({'long_name'     : 'sensor inclinometer pitch angle',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'inclination sensor embedded in anemometer sensor head; postive is anticlockwise about sonic east-west axis when viewing east to west; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['metek_InclY_Avg']              .update({'long_name'     : 'sensor inclinometer roll angle',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'inclination sensor embedded in anemometer sensor head; postive is anticlockwise about sonic south-north axis when viewing south to north; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['PTemp_Avg']                    .update({'long_name'     : 'logger electronics panel temperature',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['batt_volt_Avg']                .update({'long_name'     : 'voltage of the power source supplying the logger',
                                                            'instrument'    : 'CCampbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['counts_main_Tot']              .update({'long_name'     : 'number of completed cycles of the main scan during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_mainscan_Max']       .update({'long_name'     : 'duration of the longest scan during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_modbus_sr301_Max']   .update({'long_name'     : 'duration of the longest scan up to the first SR30 call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_modbus_sr302_Max']   .update({'long_name'     : 'duration of the longest scan up to the second SR30 call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_modbus_vaisala_Max'] .update({'long_name'     : 'duration of the longest scan up to the Vaisala call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell R1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_sdi1_Max']           .update({'long_name'     : 'duration of the longest scan up to the first SDI-12 call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_sdi2_Max']           .update({'long_name'     : 'duration of the longest scan up to the second SDI-12 call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['call_time_efoy_Max']           .update({'long_name'     : 'duration of the longest scan up to the EFOY call during the 1 min averaging interval',
                                                            'instrument'    : 'Campbell CR1000X',
                                                            'methods'       : '',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['sr30_swu_DegC_Avg']            .update({'long_name'     : 'average case temperature of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})
    
    lev1_slow_atts['sr30_swu_DegC_Std']            .update({'long_name'     : 'standard deviation of the case of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swu_Irr_Avg']             .update({'long_name'     : 'average irradiance of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'thermopile pyranometer; RS-485 protocol',
                                                            'methods'       : 'RS-485',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swu_Irr_Std']             .update({'long_name'     : 'standard deviation of the irradiance of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swu_IrrC_Avg']            .update({'long_name'     : 'average irradiance of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swu_IrrC_Std']            .update({'long_name'     : 'standard deviation of the temperature-corrected irradiance of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_DegC_Avg']            .update({'long_name'     : 'average case temperature of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_DegC_Std']            .update({'long_name'     : 'standard deviation of the case of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_Irr_Avg']             .update({'long_name'     : 'average irradiance of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_Irr_Std']             .update({'long_name'     : 'standard deviation of the irradiance of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_IrrC_Avg']            .update({'long_name'     : 'average irradiance of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_IrrC_Std']            .update({'long_name'     : 'standard deviation of the temperature-corrected irradiance of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['apogee_body_T_Avg']            .update({'long_name'     : 'average of the sensor body temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Apogee SI-4H1-SS IRT',
                                                            'methods'       : 'infrared thermometer; SDI-12 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['apogee_body_T_Std']            .update({'long_name'     : 'standard deviation of the infrared thermometer body temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Apogee SI-4H1-SS IRT',
                                                            'methods'       : 'infrared thermometer; SDI-12 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['apogee_targ_T_Avg']            .update({'long_name'     : 'average of the sensor target 8-14 micron brightness temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Apogee SI-4H1-SS IRT',
                                                            'methods'       : 'thermopile infrared thermometer; SDI-12 protocol',
                                                            'height'        : 'surface',
                                                            'location'      : boom_location,})

    lev1_slow_atts['apogee_targ_T_Std']            .update({'long_name'     : 'standard deviation of the the sensor target 8-14 micron brightness temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Apogee SI-4H1-SS IRT',
                                                            'methods'       : 'thermopile infrared thermometer; SDI-12 protocol',
                                                            'height'        : 'surface',
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr50_dist_Avg']                .update({'long_name'     : 'average of the uncorrected distance between the sensor and the surface during the 1 min averaging interval',
                                                            'instrument'    : 'SR50A',
                                                            'methods'       : 'acoustic ranger; SDI-12 protocol',
                                                            'height'        : 'surface',
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr50_dist_Std']                .update({'long_name'     : 'standard deviation of the uncorrected distance between the sensor and the surface during the 1 min averaging interval',
                                                            'instrument'    : 'SR50A',
                                                            'methods'       : 'acoustic ranger; SDI-12 protocol',
                                                            'height'        : 'surface',
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr50_qc_Avg']                  .update({'long_name'     : 'quality number of the distance measurement',
                                                            'instrument'    : 'SR50A',
                                                            'methods'       : 'acoustic ranger; SDI-12 protocol',
                                                            'height'        : 'surface',
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_RH_Avg']               .update({'long_name'     : 'average of the relative humidity during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor, heated capacitive thin-film polymer; RS-485 protocol',
                                                            'height'        : boomRh_height,
                                                            'location'      : boom_location,})
    
    lev1_slow_atts['vaisala_RH_Std']               .update({'long_name'     : 'standard deviation of the relative humidity during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor, heated capacitive thin-film polymer; RS-485 protocol',
                                                            'height'        : boomRh_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_T_Avg']                .update({'long_name'     : 'average of the air temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor, PT100 RTD; RS-485 protocol',
                                                            'height'        : boomTe_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_T_Std']                .update({'long_name'     : 'standard deviation of the air temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor, PT100 RTD; RS-485 protocol',
                                                            'height'        : boomTe_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_Td_Avg']               .update({'long_name'     : 'average of the dewpoint temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'calculated by sensor electonics; RS-485 protocol',
                                                            'height'        : boomRh_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_Td_Std']               .update({'long_name'     : 'standard deviation of the dewpoint temperature during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'calculated by sensor electronics; RS-485 protocol',
                                                            'height'        : boomRh_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_P_Avg']                .update({'long_name'     : 'average of the atmospheric pressure during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor; RS-485 protocol',
                                                            'height'        : boomPr_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['vaisala_P_Std']                .update({'long_name'     : 'standard deviation of the atmospheric pressure during the 1 min averaging interval',
                                                            'instrument'    : 'Vaisala PTU300',
                                                            'methods'       : 'meteorology sensor; RS-485 protocol',
                                                            'height'        : boomPr_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['metek_x_Avg']                  .update({'long_name'     : 'average wind velocity in x during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['metek_x_Std']                  .update({'long_name'     : 'standard deviation of the wind velocity in x during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['metek_y_Avg']                  .update({'long_name'     : 'average wind velocity in y during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['metek_y_Std']                  .update({'long_name'     : 'standard deviation of the wind velocity in y during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})
    
    lev1_slow_atts['metek_z_Avg']                  .update({'long_name'     : 'average wind velocity in z during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['metek_z_Std']                  .update({'long_name'     : 'standard deviation of the wind velocity in z during the 1 min averaging interval',
                                                            'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                            'methods'       : 'sonic anemometer, source data reported at 20 Hz; protocol RS-422',
                                                            'height'        : metek_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['ir20_lwu_mV_Avg']              .update({'long_name'     : 'average thermopile voltage of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; differential voltage',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_mV_Std']              .update({'long_name'     : 'standard deviation of the thermopile voltage of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; differential voltage',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_Case_R_Avg']          .update({'long_name'     : 'average resistance of the case thermistor of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_Case_R_Std']          .update({'long_name'     : 'standard deviation of the resistance of the case thermistor of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_DegC_Avg']            .update({'long_name'     : 'average case temperature of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor; Steinhart-Hart A=0.0010295, B=0.0002391, C=0.0000001568',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_DegC_Std']            .update({'long_name'     : 'standard deviation of the case temperature of the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor; Steinhart-Hart A=0.0010295, B=0.0002391, C=0.0000001568',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_Wm2_Avg']             .update({'long_name'     : 'average flux from the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; preliminary calibration: uses thermopile sensitvity but not temperature dependence',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_Wm2_Std']             .update({'long_name'     : 'standard deviation of the flux from the downward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; preliminary calibration: uses thermopile sensitvity but not temperature dependence',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_mV_Avg']              .update({'long_name'     : 'average thermopile voltage of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; differential voltage',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_mV_Std']              .update({'long_name'     : 'standard deviation of the thermopile voltage of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; differential voltage',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_Case_R_Avg']          .update({'long_name'     : 'average resistance of the case thermistor of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_Case_R_Std']          .update({'long_name'     : 'standard deviation of the resistance of the case thermistor of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_DegC_Avg']            .update({'long_name'     : 'average case temperature of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor; Steinhart-Hart A=0.0010295, B=0.0002391, C=0.0000001568',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_DegC_Std']            .update({'long_name'     : 'standard deviation of the case temperature of the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; analog; half-bridge, 100kOhm 0.01% ref resistor; Steinhart-Hart A=0.0010295, B=0.0002391, C=0.0000001568',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_Wm2_Avg']             .update({'long_name'     : 'average flux from the upward-facing sensor during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; preliminary calibration: uses thermopile sensitvity but not temperature dependence',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_Wm2_Std']             .update({'long_name'     : 'standard deviation of the flux from the sensor pyrgeometer during the 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux IR20',
                                                            'methods'       : 'thermopile pyrgeometer; preliminary calibration: uses thermopile sensitvity but not temperature dependence',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['fp_A_mV_Avg']                  .update({'long_name'     : 'average thermopile voltage during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_A_mV_Std']                  .update({'long_name'     : 'standard deviation of the thermopile voltage during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_A_Wm2_Avg']                 .update({'long_name'     : 'average flux during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_A_Wm2_Std']                 .update({'long_name'     : 'standard deviation of the flux during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_B_mV_Avg']                  .update({'long_name'     : 'average thermopile voltage during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_B_mV_Std']                  .update({'long_name'     : 'standard deviation of the thermopile voltage during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_B_Wm2_Avg']                 .update({'long_name'     : 'average flux during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['fp_B_Wm2_Std']                 .update({'long_name'     : 'standard deviation of the flux during 1 min averaging interval',
                                                            'instrument'    : 'Hukseflux HFP01',
                                                            'methods'       : 'thermopile conductive flux plate; analog; differential voltage',
                                                            'height'        : 'subsurface, -5 cm',
                                                            'location'      : 'soil',})

    lev1_slow_atts['licor_T_Avg']                  .update({'long_name'     : 'temperature at strut',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_co2_Avg']                .update({'long_name'     : 'average CO2 gas density during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_co2_Std']                .update({'long_name'     : 'standard deviation of the CO2 gas density during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_h2o_Avg']                .update({'long_name'     : 'average water vapor density during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_h2o_Std']                .update({'long_name'     : 'standard deviation of the water vapor density during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_co2_str_out_Avg']        .update({'long_name'     : 'average CO2 signal strength diagnostic during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['licor_co2_str_out_Std']        .update({'long_name'     : 'standard deviation of the CO2 signal strength diagnostic during 1 min averaging interval',
                                                            'instrument'    : 'Licor 7500-DS',
                                                            'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz; protocol TCP/IP',
                                                            'height'        : licor_height,
                                                            'location'      : mast_location,})

    lev1_slow_atts['sr30_swu_fantach_Avg']         .update({'long_name'     : 'average fan speed of the downward-facing (SWU) pyranometer during 1 min averaging period',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case fan; RS-485',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swu_heatA_Avg']           .update({'long_name'     : 'average case heating current of the downward-facing sensor during 1 min averaging period',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case heater; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_fantach_Avg']         .update({'long_name'     : 'average fan speed of the upward-facing sensor during 1 min averaging period',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case fan; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_heatA_Avg']           .update({'long_name'     : 'average case heating current of the upward-facing sensor during 1 min averaging period',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case heater; RS-485 protocol',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwu_fan_Avg']             .update({'long_name'     : 'average fan voltage status signal of the downward-facing ventilator during the 1 min averaging period',
                                                            'instrument'    : 'Hukseflux VU01',
                                                            'methods'       : 'ventilator fan; analog',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['ir20_lwd_fan_Avg']             .update({'long_name'     : 'average fan voltage status signal of the upward-facing ventilator during the 1 min averaging period',
                                                            'instrument'    : 'Hukseflux VU01',
                                                            'methods'       : 'ventilator fan; analog',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['efoy_Error_Max']               .update({'long_name'     : 'error code reported by the EFOY',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_FuellSt_Avg']             .update({'long_name'     : 'mixing tank fluid level',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_Ubat_Avg']                .update({'long_name'     : 'battery voltage',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_Laus_Avg']                .update({'long_name'     : 'battery charging current',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_Tst_Avg']                 .update({'long_name'     : 'stack temperature',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_Tint_Avg']                .update({'long_name'     : 'internal temperature',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['efoy_Twt_Avg']                 .update({'long_name'     : 'heat exchanger temperature',
                                                            'instrument'    : 'EFOY',
                                                            'methods'       : 'Direct Methanol Fuel Cell (DFMC) power supply; MODBUS RS-232 protocol',
                                                            'height'        : 'N/A',
                                                            'location'      : 'logger box',})

    lev1_slow_atts['sr30_swu_tilt_Avg']            .update({'long_name'     : 'horizontal tilt of the downward-facing sensor',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case level; RS-485 protocol; level defined as 180 deg',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})

    lev1_slow_atts['sr30_swd_tilt_Avg']            .update({'long_name'     : 'horizontal tilt of the upward-facing sensor',
                                                            'instrument'    : 'Hukseflux SR30',
                                                            'methods'       : 'thermopile pyranometer case level; RS-485 protocol; level defined as 0 deg',
                                                            'height'        : boom_height,
                                                            'location'      : boom_location,})


    return lev1_slow_atts, list(lev1_slow_atts.keys()).copy()

def define_turb_variables():
        
    turb_atts = OrderedDict()

    turb_atts['Hs']            = {'units' : 'W/m2'}
    turb_atts['Hl']            = {'units' : 'W/m2'}
    turb_atts['CO2_flux']      = {'units' : 'mg*m^-2*s^-1'}
    turb_atts['CO2_flux_Wm2']  = {'units' : 'W/m2'}
    turb_atts['Cd']            = {'units' : 'dimensionless'}
    turb_atts['ustar']         = {'units' : 'm/s'}
    turb_atts['Tstar']         = {'units' : 'degC'}
    turb_atts['zeta_level_n']  = {'units' : 'dimensionless'}
    turb_atts['WU_csp']        = {'units' : 'm^2/s^2'}
    turb_atts['WV_csp']        = {'units' : 'm^2/s^2'}
    turb_atts['UV_csp']        = {'units' : 'm^2/s^2'}
    turb_atts['WT_csp']        = {'units' : 'degC*m/s'}
    turb_atts['UT_csp']        = {'units' : 'degC*m/s'}
    turb_atts['VT_csp']        = {'units' : 'degC*m/s'}
    turb_atts['Wq_csp']        = {'units' : 'm/s*kg/kg'}
    turb_atts['Uq_csp']        = {'units' : 'm/s*kg/kg'}
    turb_atts['Vq_csp']        = {'units' : 'm/s*kg/kg'}
    turb_atts['Wc_csp']        = {'units' : 'm*s^-1*mg*m^-2*s^-1'}
    turb_atts['Uc_csp']        = {'units' : 'm*s^-1*mg*m^-2*s^-1'}
    turb_atts['Vc_csp']        = {'units' : 'm*s^-1*mg*m^-2*s^-1'}
    turb_atts['phi_U']         = {'units' : 'dimensionless'}
    turb_atts['phi_V']         = {'units' : 'dimensionless'}
    turb_atts['phi_W']         = {'units' : 'dimensionless'}
    turb_atts['phi_T']         = {'units' : 'dimensionless'}
    turb_atts['phi_UT']        = {'units' : 'dimensionless'}
    turb_atts['nSU']           = {'units' : 'Power/Hz'}
    turb_atts['nSV']           = {'units' : 'Power/Hz'}
    turb_atts['nSW']           = {'units' : 'Power/Hz'}
    turb_atts['nST']           = {'units' : 'Power/Hz'}
    turb_atts['nSq']           = {'units' : 'Power/Hz'}
    turb_atts['nSc']           = {'units' : 'Power/Hz'}
    turb_atts['epsilon_U']     = {'units' : 'm^2/s^3'}
    turb_atts['epsilon_V']     = {'units' : 'm^2/s^3'}
    turb_atts['epsilon_W']     = {'units' : 'm^2/s^3'}
    turb_atts['epsilon']       = {'units' : 'm^2/s^3'}
    turb_atts['Phi_epsilon']   = {'units' : 'dimensionless'}
    turb_atts['NT']            = {'units' : 'degC^2/s'}
    turb_atts['Phi_NT']        = {'units' : 'dimensionless'}
    turb_atts['Phix']          = {'units' : 'deg'}
    turb_atts['DeltaU']        = {'units' : 'm/s'}
    turb_atts['DeltaV']        = {'units' : 'm/s'}
    turb_atts['DeltaT']        = {'units' : 'degC'}
    turb_atts['Deltaq']        = {'units' : 'kg/kg'}
    turb_atts['Deltac']        = {'units' : 'mg/m3'}
    turb_atts['sigU']          = {'units' : 'm/s'}
    turb_atts['sigV']          = {'units' : 'm/s'}
    turb_atts['sigW']          = {'units' : 'm/s'}
    turb_atts['fs']            = {'units' : 'Hz'}
    turb_atts['dfs']           = {'units' : 'Hz'}
    turb_atts['sUs']           = {'units' : '(m/s)^2/Hz'}
    turb_atts['sVs']           = {'units' : '(m/s)^2/Hz'}
    turb_atts['sWs']           = {'units' : '(m/s)^2/Hz'}
    turb_atts['sTs']           = {'units' : 'degC^2/Hz'}
    turb_atts['sqs']           = {'units' : '(kg/m3)/Hz'}
    turb_atts['scs']           = {'units' : '(mg m^-2 s^-1)^2/Hz'}
    turb_atts['cWUs']          = {'units' : '(m/s)^2/Hz'}
    turb_atts['cWVs']          = {'units' : '(m/s)^2/Hz'}
    turb_atts['cUVs']          = {'units' : '(m/s)^2/Hz'}
    turb_atts['cWTs']          = {'units' : '(m/s*degC)/Hz'}
    turb_atts['cUTs']          = {'units' : '(m/s*degC)/Hz'}
    turb_atts['cVTs']          = {'units' : '(m/s*degC)/Hz'}
    turb_atts['cWqs']          = {'units' : '(m/s*kg/kg)/Hz'}
    turb_atts['cUqs']          = {'units' : '(m/s*kg/kg)/Hz'}
    turb_atts['cVqs']          = {'units' : '(m/s*kg/kg)/Hz'}
    turb_atts['cWcs']          = {'units' : '(m/s*mg*m^-2*s^-1)/Hz'}
    turb_atts['cUcs']          = {'units' : '(m/s*mg*m^-2*s^-1)/Hz'}
    turb_atts['cVcs']          = {'units' : '(m/s*mg*m^-2*s^-1)/Hz'}
    turb_atts['bulk_Hs']       = {'units' : 'W/m2'}
    turb_atts['bulk_Hl']       = {'units' : 'W/m2'}
    turb_atts['bulk_tau']      = {'units' : 'Pa'}
    turb_atts['bulk_z0']       = {'units' : 'm'}
    turb_atts['bulk_z0t']      = {'units' : 'm'}
    turb_atts['bulk_z0q']      = {'units' : 'm'}
    turb_atts['bulk_L']        = {'units' : 'm'}
    turb_atts['bulk_ustar']    = {'units' : 'm/s'}
    turb_atts['bulk_tstar']    = {'units' : 'K'}
    turb_atts['bulk_qstar']    = {'units' : 'kg/kg'}
    turb_atts['bulk_dter']     = {'units' : 'degC'}
    turb_atts['bulk_dqer']     = {'units' : 'kg/kg'}
    turb_atts['bulk_Cd']       = {'units' : 'unitless'}
    turb_atts['bulk_Ch']       = {'units' : 'unitless'}
    turb_atts['bulk_Ce']       = {'units' : 'unitless'}
    turb_atts['bulk_Cdn_10m']  = {'units' : 'unitless'}
    turb_atts['bulk_Chn_10m']  = {'units' : 'unitless'}
    turb_atts['bulk_Cen_10m']  = {'units' : 'unitless'}
    turb_atts['bulk_Rr']       = {'units' : 'unitless'}
    turb_atts['bulk_Rt']       = {'units' : 'unitless'}
    turb_atts['bulk_Rq']       = {'units' : 'unitless'}


    # !! The turbulence data. A lot of it... Almost wonder if this metadata should reside in a separate
    # file?  Some variables are dimensionless, meaning they are both scaled & unitless and the result is
    # independent of height such that it sounds a little funny to have a var name like
    # MO_dimensionless_param_2_m, but that is the only way to distinguish between the 4 calculations. So,
    # for these I added ", calculated from 2 m" or whatever into the long_name.  Maybe there is a better
    # way?

    turb_atts['Hs']              .update({'long_name'  : 'sensible heat flux',
                                          'cf_name'    : 'upward_sensible_heat_flux_in_air',
                                          'instrument' : 'Metek uSonic-Cage MP sonic anemometer',
                                          'methods'    : 'source data was 20 Hz samples averged to 10 Hz. Calculation by eddy covariance using sonic temperature based on integration of the wT-covariance spectrum',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Hl']              .update({'long_name'  : 'latent heat flux',
                                          'cf_name'    : 'upward_latent_heat_flux_in_air',
                                          'instrument' : 'Metek uSonic-Cage MP sonic anemometer, Licor 7500 DS',
                                          'methods'    : 'source data was 20 Hz samples averged to 10 Hz. Calculation by eddy covariance using sonic temperature based on integration of the wT-covariance spectrum. Source h2o data was vapor density (g/m3). No WPL correction applied.',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['CO2_flux']        .update({'long_name'  : 'co2 mass flux',
                                          'cf_name'    : '',
                                          'instrument' : 'Metek uSonic-Cage MP sonic anemometer, Licor 7500 DS',
                                          'methods'    : 'source data was 20 Hz samples averged to 10 Hz. Calculation by eddy covariance using sonic temperature based on integration of the wT-covariance spectrum. Source co2 data was co2 density (mmol/m3). No WPL correction applied.',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['CO2_flux_Wm2']    .update({'long_name'  : 'co2 photosynthetic storage term (Webb corrected)',
                                          'cf_name'    : '',
                                          'instrument' : 'Metek uSonic-Cage MP sonic anemometer, Licor 7500 DS',
                                          'methods'    : 'source data was 20 Hz samples averged to 10 Hz. Calculation by eddy covariance using sonic temperature based on integration of the wT-covariance spectrum. Source co2 data was co2 density (mmol/m3). No WPL correction applied. Conversion follows Grachev et al. (2020) https://doi.org/10.1016/j.agrformet.2019.107823',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})
    
    turb_atts['Cd']              .update({'long_name'  : 'Drag coefficient based on the momentum flux, calculated from 2 m',
                                          'cf_name'    : '',
                                          'height'     : 'N/A',
                                          'location'   : mast_location,})

    turb_atts['ustar']           .update({'long_name'  : 'friction velocity (based only on the downstream, uw, stress components)',
                                          'cf_name'    : '',
                                          'height'     : 'N/A',
                                          'location'   : mast_location,})

    turb_atts['Tstar']           .update({'long_name'  : 'temperature scale',
                                          'cf_name'    : '',
                                          'height'     : 'N/A',
                                          'location'   : mast_location,})

    turb_atts['zeta_level_n']    .update({'long_name'  : 'Monin-Obukhov stability parameter, z/L, calculated from 2 m',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['WU_csp']          .update({'long_name'  : 'WU-covariance based on the WU-cospectra integration; W and U are the vertical-orthogonal (to stream) and streamwise wind vectors',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['WV_csp']          .update({'long_name'  : 'WV-covariance based on the wv-cospectra integration; W and V are the vertical-orthogonal (to stream) and cross-stream wind vectors',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['UV_csp']          .update({'long_name'  : 'UV-covariance based on the uv-cospectra integration; U and V are the streamwise and cross-stream wind vectors',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['WT_csp']          .update({'long_name'  : 'WT-covariance, vertical flux of the sonic temperature; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['UT_csp']          .update({'long_name'  : 'UT-covariance, vertical flux of the sonic temperature; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['VT_csp']          .update({'long_name'  : 'VT-covariance, vertical flux of the sonic temperature; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Wq_csp']          .update({'long_name'  : 'Wq-covariance, vertical flux of h2o; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Uq_csp']          .update({'long_name'  : 'Uq-covariance, vertical flux of h2o; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Vq_csp']          .update({'long_name'  : 'Vq-covariance, vertical flux of h2o; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Wc_csp']          .update({'long_name'  : 'wc-covariance, vertical flux of co2; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Uc_csp']          .update({'long_name'  : 'uc-covariance, vertical flux of co2; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Vc_csp']          .update({'long_name'  : 'vc-covariance, vertical flux of co2; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['phi_U']           .update({'long_name'  : 'MO universal function for the standard deviations, calculated from 2 m; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['phi_V']           .update({'long_name'  : 'MO universal function for the standard deviations, calculated from 2 m; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['phi_W']           .update({'long_name'  : 'MO universal function for the standard deviations, calculated from 2 m; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['phi_T']           .update({'long_name'  : 'MO universal function for the standard deviations, calculated from 2 m',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['phi_UT']          .update({'long_name'  : 'MO universal function for the horizontal heat flux, calculated from 2 m; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['epsilon_U']       .update({'long_name'  : 'Dissipation rate of the turbulent kinetic energy in u based on the energy spectra of the longitudinal velocity component in the inertial subrange; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['epsilon_V']       .update({'long_name'  : 'Dissipation rate of the turbulent kinetic energy in v based on the energy spectra of the longitudinal velocity component in the inertial subrange; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['epsilon_W']       .update({'long_name'  : 'Dissipation rate of the turbulent kinetic energy in w based on the energy spectra of the longitudinal velocity component in the inertial subrange; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['epsilon']         .update({'long_name'  : 'Dissipation rate of the turbulent kinetic energy = median of the values derived from U, V, & W',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Phi_epsilon']     .update({'long_name'  : 'Monin-Obukhov universal function Phi_epsilon based on the median epsilon, calculated from 2 m',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['nSU']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of U; U is the streamwise wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['nSV']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of V; V is the cross-stream wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['nSW']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of W; W is the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['nST']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of sonic temperature',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['nSq']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of q',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['nSc']             .update({'long_name'  : 'Median spectral slope in the inertial subrange of co2',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['NT']              .update({'long_name'  : 'The dissipation (destruction) rate for half the temperature variance',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Phi_NT']          .update({'long_name'  : 'Monin-Obukhov universal function Phi_Nt, calculated from 2 m',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Phix']            .update({'long_name'  : 'Angle of attack',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['DeltaU']          .update({'long_name'  : 'Stationarity diagnostic: Steadiness of the streamwise wind vector (trend)',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['DeltaV']          .update({'long_name'  : 'Stationarity diagnostic: Steadiness of the cross-stream wind vector (trend)',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['DeltaT']          .update({'long_name'  : 'Stationarity diagnostic: Steadiness of the sonic temperature (trend)',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['Deltaq']          .update({'long_name'  : 'Stationarity diagnostic: Steadiness of q (trend)',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['Deltac']          .update({'long_name'  : 'Stationarity diagnostic: Steadiness of co2 (trend)',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})
        
    turb_atts['sigU' ]           .update({'long_name'  :'Standard deviation of streamwise wind vector',
                                          'cf_name'    :'',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})
        
    turb_atts['sigV' ]           .update({'long_name'  :'Standard deviation of cross-stream wind vector',
                                          'cf_name'    :'',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['sigW' ]           .update({'long_name'  :'Standard deviation of the vertical-orthogonal (to stream) wind vector',
                                          'cf_name'    :'',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})   

    turb_atts['fs']              .update({'long_name'  : 'frequency',
                                          'cf_name'    : '',
                                          'height'     : 'N/A',
                                          'location'   : 'N/A',})

    turb_atts['dfs']             .update({'long_name'  : 'smoothed 1st derivative of frequency vector, fs',
                                          'cf_name'    : '',
                                          'height'     : 'N/A',
                                          'location'   : 'N/A',})

    turb_atts['sUs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of the streamwise wind vector on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['sVs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of the cross-stream wind vector on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['sWs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of the vertical-orthogonal (to stream) wind vector on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['sTs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of sonic temperature on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['sqs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of q on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['scs']             .update({'long_name'  : 'smoothed power spectral density (Welch) of co2 on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cWUs']            .update({'long_name'  : 'smoothed co-spectral density between the vertical-orthogonal (to stream) and streamwise wind vectors on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cWVs']            .update({'long_name'  : 'smoothed co-spectral density between the vertical-orthogonal (to stream) and cross-stream wind vectors on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cUVs']            .update({'long_name'  : 'smoothed co-spectral density between the streamwise and cross-stream wind vectors on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cWTs']            .update({'long_name'  : 'smoothed co-spectral density between the vertical-orthogonal (to stream) wind vector and sonic temperature on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cUTs']            .update({'long_name'  : 'smoothed co-spectral density between the streamwise wind vector and sonic temperature on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cVTs']            .update({'long_name'  : 'smoothed co-spectral density between the cross-stream wind vector and sonic temperature on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['cWqs']            .update({'long_name'  : 'smoothed co-spectral density between the vertical-orthogonal (to stream) wind vector and q on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cUqs']            .update({'long_name'  : 'smoothed co-spectral density between the streamwise wind vector and q on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cVqs']            .update({'long_name'  : 'smoothed co-spectral density between the cross-wind wind vector and q on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cWcs']            .update({'long_name'  : 'smoothed co-spectral density between the vertical-orthogonal (to stream) wind vector and co2 on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cUcs']            .update({'long_name'  : 'smoothed co-spectral density between the streamwise wind vector and co2 on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['cVcs']            .update({'long_name'  : 'smoothed co-spectral density between the cross-stream wind vector and co2 on frequency, fs',
                                          'cf_name'    : '',
                                          'height'     : licor_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_Hs']         .update({'long_name'  : 'sensible heat flux',
                                          'cf_name'    : 'upward_sensible_heat_flux_in_air',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomTe_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_Hl']         .update({'long_name'  : 'latent heat flux',
                                          'cf_name'    : 'upward_latent_heat_flux_in_air',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomRh_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_tau']        .update({'long_name'  : 'wind stress',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_z0']          .update({'long_name' : 'roughness length',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_z0t']         .update({'long_name' : 'roughness length, temperature',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_z0q']         .update({'long_name' : 'roughness length, humidity',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomRh_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_L']           .update({'long_name' : 'Obukhov length',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : 'N/A',
                                          'location'   : mast_location,})

    turb_atts['bulk_ustar']       .update({'long_name' : 'friction velocity (sqrt(momentum flux)), ustar',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : 'N/A',
                                          'location'   : mast_location,})

    turb_atts['bulk_tstar']       .update({'long_name' : 'temperature scale, tstar',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : 'N/A',
                                          'location'   : boom_location,})

    turb_atts['bulk_qstar']       .update({'long_name' : 'specific humidity scale, qstar ',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : 'N/A',
                                          'location'   : boom_location,})

    turb_atts['bulk_dter']        .update({'long_name' : 'diagnostic',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomTe_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_dqer']        .update({'long_name' : 'diagnostic',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomRh_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_Cd']          .update({'long_name' : 'transfer coefficient for stress',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_Ch']          .update({'long_name' : 'transfer coefficient for Hs',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_Ce']          .update({'long_name' : 'transfer coefficient for Hl',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : boomRh_height,
                                          'location'   : boom_location,})

    turb_atts['bulk_Cdn_10m']     .update({'long_name' : '10 m neutral transfer coefficient for stress',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : '10 m',
                                          'location'   : mast_location,})

    turb_atts['bulk_Chn_10m']     .update({'long_name' : '10 m neutral transfer coefficient for Hs',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : '10 m',
                                          'location'   : mast_location,})

    turb_atts['bulk_Cen_10m']     .update({'long_name' : '10 m neutral transfer coefficient for Hl',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : '10 m',
                                          'location'   : mast_location,})

    turb_atts['bulk_Rr']          .update({'long_name' : 'roughness Reynolds number for velocity',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_Rt']          .update({'long_name' : 'roughness Reynolds number for temperature',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    turb_atts['bulk_Rq']          .update({'long_name' : 'roughness Reynolds number for humidity',
                                          'cf_name'    : '',
                                          'instrument' : 'various',
                                          'methods'    : 'Bulk calc. Fairall et al. (1996) https://doi.org/10.1029/95JC03205, Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm',
                                          'height'     : metek_height,
                                          'location'   : mast_location,})

    return turb_atts, list(turb_atts.keys()).copy()


# defines the column names of the logger data for the flux stations and thus also the output variable names in
# the level1 data files that are supposed to be "nearly-raw netcdf records". this returns two things. 1) a list of dictionaries
# where the key is the variable name and dictionary is a list of netcdf attributes. 2) a list of variable names
def define_level1_fast():

    # !!!! I know the licor order is probably wrong... but moving on so that I can finish this
    # I can't figure out why there are less columns in the fast files than I would expect given
    # the definitions in the CR1X files...
    lev1_fast_atts = OrderedDict()

    # units defined here, other properties defined in 'update' call below
    lev1_fast_atts['TIMESTAMP']           = {'units' : 'time'    }
    lev1_fast_atts['metek_x']             = {'units' : 'm/s'     }
    lev1_fast_atts['metek_y']             = {'units' : 'm/s'     }
    lev1_fast_atts['metek_z']             = {'units' : 'm/s'     }
    lev1_fast_atts['metek_T']             = {'units' : 'C'       }
    lev1_fast_atts['metek_heatstatus']    = {'units' : 'int'     }
    lev1_fast_atts['metek_senspathstate'] = {'units' : 'int'     }
    lev1_fast_atts['licor_co2']           = {'units' : 'g/m3'    }
    lev1_fast_atts['licor_h2o']           = {'units' : 'mg/m3'   }
    lev1_fast_atts['licor_pr']            = {'units' : 'hPa'     }
    lev1_fast_atts['licor_diag']          = {'units' : 'int'     }

    lev1_fast_atts['TIMESTAMP']           .update({'long_name'     : 'time of measurement',
                                                   'instrument'    : 'CR1000X',
                                                   'methods'       : 'synched to GPS',
                                                   'height'        : 'N/A',
                                                   'location'      : 'logger box',})

    lev1_fast_atts['metek_x']             .update({'long_name'     : 'wind velocity in x',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['metek_y']             .update({'long_name'     : 'wind velocity in y',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['metek_z']             .update({'long_name'     : 'wind velocity in z',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['metek_T']             .update({'long_name'     : 'acoustic temperature',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['metek_heatstatus']    .update({'long_name'     : 'transducer heating status code',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['metek_senspathstate'] .update({'long_name'     : 'number (of 9) of unusable paths',
                                                   'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                   'methods'       : 'sonic anemometer; data reported at 20 Hz; protocol RS-422',
                                                   'height'        : metek_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['licor_diag']          .update({'long_name'     : 'bit-packed diagnostic integer',
                                                   'instrument'    : 'Licor 7500-DS',
                                                   'methods'       : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol; bits 0-3 = signal strength; bit 5 = PLL; bit 6 = detector temp; bit 7 = chopper temp',
                                                   'height'        : licor_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['licor_co2']           .update({'long_name'     : 'CO2 gas density',
                                                   'instrument'    : 'Licor 7500-DS',
                                                   'methods'       : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol',
                                                   'height'        : licor_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['licor_h2o']           .update({'long_name'     : 'Water vapor density',
                                                   'instrument'    : 'Licor 7500-DS',
                                                   'methods'       : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol',
                                                   'height'        : licor_height,
                                                   'location'      : mast_location,})

    lev1_fast_atts['licor_pr']            .update({'long_name'     : 'air pressure',
                                                   'instrument'    : 'Licor 7500-DS',
                                                   'methods'       : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol; air pressure measured in electronics box',
                                                   'height'        : licor_height,
                                                   'location'      : mast_location,})

    return lev1_fast_atts, list(lev1_fast_atts.keys()).copy()

def define_level2_variables(station_name):

    if station_name == 'asfs30':
        soil_5cm_depth  = '-5 cm'
        soil_10cm_depth = '-10 cm'
        soil_20cm_depth = '-20 cm'
        soil_30cm_depth = '-30 cm'
        soil_40cm_depth = '-40 cm'
        soil_50cm_depth = '-50 cm'
    elif station_name == 'asfs50':
        soil_5cm_depth  = '-2 cm'
        soil_10cm_depth = '-7 cm'
        soil_20cm_depth = '-17 cm'
        soil_30cm_depth = '-27 cm'
        soil_40cm_depth = '-37 cm'
        soil_50cm_depth = '-47 cm'

    lev2_atts = OrderedDict()

    lev2_atts['zenith_true']             = {'units' : 'degrees'}
    lev2_atts['azimuth']                 = {'units' : 'degrees'}
    lev2_atts['sr50_dist']               = {'units' : 'meters'}
    lev2_atts['snow_depth']              = {'units' : 'cm'}
    lev2_atts['snow_flag']               = {'units' : 'unitless'}
    lev2_atts['atmos_pressure']          = {'units' : 'hPa'}
    lev2_atts['temp']                    = {'units' : 'deg C'}
    lev2_atts['rh']                      = {'units' : 'percent'}
    lev2_atts['dew_point']               = {'units' : 'deg C'}
    lev2_atts['mixing_ratio']            = {'units' : 'g/kg'}
    lev2_atts['vapor_pressure']          = {'units' : 'Pa'}
    lev2_atts['rhi']                     = {'units' : 'percent'}
    lev2_atts['brightness_temp_surface'] = {'units' : 'deg C'}
    lev2_atts['skin_temp_surface']       = {'units' : 'deg C'}
    lev2_atts['subsurface_heat_flux_A']  = {'units' : 'W/m2'}
    lev2_atts['subsurface_heat_flux_B']  = {'units' : 'W/m2'}
    lev2_atts['wspd_u_mean']             = {'units' : 'm/s'}
    lev2_atts['wspd_v_mean']             = {'units' : 'm/s'}
    lev2_atts['wspd_w_mean']             = {'units' : 'm/s'}
    lev2_atts['wspd_vec_mean']           = {'units' : 'm/s'}
    lev2_atts['wdir_vec_mean']           = {'units' : 'degrees'}
    lev2_atts['wspd_u_std']              = {'units' : 'm/s'}
    lev2_atts['wspd_v_std']              = {'units' : 'm/s'}
    lev2_atts['wspd_w_std']              = {'units' : 'm/s'}
    lev2_atts['h2o_licor']               = {'units' : 'g/m3'}
    lev2_atts['co2_licor']               = {'units' : 'mg/m3'}
    lev2_atts['down_long_hemisp']        = {'units' : 'W/m2'}
    lev2_atts['down_short_hemisp']       = {'units' : 'W/m2'}
    lev2_atts['up_long_hemisp']          = {'units' : 'W/m2'}
    lev2_atts['up_short_hemisp']         = {'units' : 'W/m2'}
    lev2_atts['s_soil']                  = {'units' : 'W/m2'}
    lev2_atts['c_soil']                  = {'units' : 'W/m2'}
    lev2_atts['k_eff_soil']              = {'units' : 'W/m/K'}
    lev2_atts['soil_vwc_5cm']            = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_5cm']             = {'units' : 'unitless'}
    lev2_atts['soil_t_5cm']              = {'units' : 'deg'}
    lev2_atts['soil_ec_5cm']             = {'units' : 'dS/m'}
    lev2_atts['soil_vwc_10cm']           = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_10cm']            = {'units' : 'unitless'}
    lev2_atts['soil_t_10cm']             = {'units' : 'deg'}
    lev2_atts['soil_ec_10cm']            = {'units' : 'dS/m'}
    lev2_atts['soil_vwc_20cm']           = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_20cm']            = {'units' : 'unitless'}
    lev2_atts['soil_t_20cm']             = {'units' : 'deg'}
    lev2_atts['soil_ec_20cm']            = {'units' : 'dS/m'}
    lev2_atts['soil_vwc_30cm']           = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_30cm']            = {'units' : 'unitless'}
    lev2_atts['soil_t_30cm']             = {'units' : 'deg'}
    lev2_atts['soil_ec_30cm']            = {'units' : 'dS/m'}
    lev2_atts['soil_vwc_40cm']           = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_40cm']            = {'units' : 'unitless'}
    lev2_atts['soil_t_40cm']             = {'units' : 'deg'}
    lev2_atts['soil_ec_40cm']            = {'units' : 'dS/m'}
    lev2_atts['soil_vwc_50cm']           = {'units' : 'm3/m3'}
    lev2_atts['soil_ka_50cm']            = {'units' : 'unitless'}
    lev2_atts['soil_t_50cm']             = {'units' : 'deg'}
    lev2_atts['soil_ec_50cm']            = {'units' : 'dS/m'}

    # add everything else to the variable NetCDF attributes
    # #########################################################################################################
    lev2_atts['zenith_true']           .update({'long_name'     : 'true solar zenith angle',
                                                'cf_name'       : '',
                                                'instrument'    : 'Hemisphere V102',
                                                'methods'       : 'Reda and Andreas, Solar position algorithm for solar radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['azimuth']               .update({'long_name'     : 'apparent solar azimuth angle',
                                                'cf_name'       : '',
                                                'instrument'    : 'Hemisphere V102',
                                                'methods'       : 'Reda and Andreas, Solar position algorithm for solar radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['sr50_dist']             .update({'long_name'     : 'distance to surface from SR50; temperature compensation correction applied',
                                                'cf_name'       : '',
                                                'instrument'    : 'Campbell Scientific SR50A',
                                                'methods'       : 'unheated, temperature correction applied',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['snow_depth']            .update({'long_name'     : 'snow depth near station base',
                                                'cf_name'       : 'surface_snow_thickness',
                                                'instrument'    : 'Campbell Scientific SR50A',
                                                'methods'       : 'derived snow depth from temperature-corrected SR50 distance values based on initialization. footprint nominally 0.47 m radius.',
                                                'comment'       : 'Increasing values suggest snowfall, decreasing values suggest snow loss through redistribution or melt. Negative values do occur in the dataset when there has been melt of the snow/ice surface. Please read the associated data paper for more information.',
                                                'height'        : 'surface',
                                                'location'      : boom_location,})

    lev2_atts['snow_flag']             .update({'long_name'     : 'flag for snow cover',
                                                'cf_name'       : '',
                                                'instrument'    : '',
                                                'methods'       : 'Based on daily mean albedo using a threshold of 0.3 following Stone et al. (2002) JGR 107(D10)',
                                                'comment'       : 'Resolution is daily. 1 = snow cover, 0 = snow free',
                                                'height'        : 'surface',
                                                'location'      : '',})
    
    lev2_atts['atmos_pressure']        .update({'long_name'     : 'atmospheric pressure',
                                                'cf_name'       : 'air_pressure',
                                                'instrument'    : 'Vaisala PTU 300',
                                                'methods'       : 'digitally polled from instument',
                                                'height'        : boomPr_height,
                                                'location'      : boom_location,})

    lev2_atts['temp']                  .update({'long_name'     : 'air temperature',
                                                'cf_name'       : 'air_temperature',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'digitally polled from instument',
                                                'height'        : boomTe_height,
                                                'location'      : boom_location,})

    lev2_atts['rh']                    .update({'long_name'     : 'relative humidity',
                                                'cf_name'       : 'relative_humidity',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'digitally polled from instument',
                                                'height'        : boomRh_height,
                                                'location'      : boom_location,})

    lev2_atts['dew_point']               .update({'long_name'   : 'dewpoint temperature',
                                                'cf_name'       : 'dew_point_temperature',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'digitally polled from instument',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['mixing_ratio']            .update({'long_name'   : 'mixing ratio',
                                                'cf_name'       : 'humidity_mixing_ratio',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'calculated from measured variables following Wexler (1976)',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['vapor_pressure']          .update({'long_name'   : 'vapor pressure',
                                                'cf_name'       : '',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'calculated from measured variables following Wexler (1976)',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})
 
    lev2_atts['rhi']                     .update({'long_name'   : 'relative humidity wrt ice',
                                                'cf_name'       : '',
                                                'instrument'    : 'Vaisala PTU300',
                                                'methods'       : 'calculated from measured variables following Hyland & Wexler (1983)',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['brightness_temp_surface'] .update({'long_name'   : 'sensor target 8-14 micron brightness temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'Apogee SI-4H1-SS IRT',
                                                'methods'       : 'digitally polled from instument. No emissivity correction. No correction for reflected incident.',
                                                'height'        : 'surface',
                                                'location'      : boom_location,})

    lev2_atts['skin_temp_surface']      .update({'long_name'    : 'surface radiometric skin temperature, assummed emissivity, corrected for IR reflection',
                                                'cf_name'       : '',
                                                'instrument'    : 'IR20 LWu, LWd',
                                                'methods'       : 'Eq.(2.2) Persson et al. (2002) https://www.doi.org/10.1029/2000JC000705; emis = 0.985',
                                                'height'        : 'surface',
                                                'location'      : boom_location,})
    
    lev2_atts['subsurface_heat_flux_A'] .update({'long_name'    : 'conductive flux from plate A, defined positive upward',
                                                'cf_name'       : '',
                                                'instrument'    : 'Hukseflux HFP01',
                                                'methods'       : 'Sensitivity 63.00/1000 [mV/(W/m2)]',
                                                'height'        : 'subsurface, -5 cm',
                                                'location'      : 'soil',})
    
    lev2_atts['subsurface_heat_flux_B'] .update({'long_name'    : 'conductive flux from plate B, defined positive upward',
                                                'cf_name'       : '', 
                                                'instrument'    : 'Hukseflux HFP01',
                                                'methods'       : 'Sensitivity 63.91/1000 [mV/(W/m2)]',
                                                'height'        : 'subsurface, -5 cm',
                                                'location'      : 'soil',})
        
    lev2_atts['wspd_u_mean']           .update({'long_name'     : 'Metek u-component',
                                                'cf_name'       : 'eastward_wind',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'v defined positive west-to-east',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
        
    lev2_atts['wspd_v_mean']           .update({'long_name'     : 'Metek v-component',
                                                'cf_name'       : 'northward_wind',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'v defined positive south-to-north',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
                                                
    lev2_atts['wspd_w_mean']           .update({'long_name'     : 'Metek w-component',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'w defined positive up',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})

    lev2_atts['wspd_vec_mean']         .update({'long_name'     : 'average metek wind speed',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
        
    lev2_atts['wdir_vec_mean']         .update({'long_name'     : 'average metek wind direction',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})    
        
    lev2_atts['wspd_u_std']           .update({ 'long_name'     : 'u metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,}) 
    
    lev2_atts['wspd_v_std']           .update({ 'long_name'     : 'v metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
    
    lev2_atts['wspd_w_std']           .update({ 'long_name'     : 'w metek obs standard deviation',
                                                'cf_name'       : '',
                                                'instrument'    : 'Metek uSonic-Cage MP sonic anemometer',
                                                'methods'       : 'derived from horizontal wind components after coordinate transformation from body to earth reference frame',
                                                'height'        : metek_height,
                                                'location'      : mast_location,})
    
    lev2_atts['h2o_licor']            .update({ 'long_name'     : 'Licor water vapor mass density',
                                                'cf_name'       : '',
                                                'instrument'    : 'Licor 7500-DS',
                                                'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz',
                                                'height'        : licor_height,
                                                'location'      : mast_location,})

    lev2_atts['co2_licor']            .update({ 'long_name'     : 'Licor CO2 gas density',
                                                'cf_name'       : '',
                                                'instrument'    : 'Licor 7500-DS',
                                                'methods'       : 'open-path optical gas analyzer, source data reported at 20 Hz',
                                                'height'        : licor_height,
                                                'location'      : mast_location,})

    lev2_atts['down_long_hemisp']     .update({ 'long_name'     : 'net downward longwave flux',
                                                'cf_name'       : 'surface_net_downward_longwave_flux',
                                                'instrument'    : 'Hukseflux IR20 pyrgeometer',
                                                'methods'       : 'hemispheric longwave radiation',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['down_short_hemisp']    .update({ 'long_name'     : 'net downward shortwave flux',
                                                'cf_name'       : 'surface_net_downward_shortwave_flux',
                                                'instrument'    : 'Hukseflux SR30 pyranometer',
                                                'methods'       : 'hemispheric shortwave radiation',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['up_long_hemisp']       .update({ 'long_name'     : 'net upward longwave flux',
                                                'cf_name'       : 'surface_net_upward_longwave_flux',
                                                'instrument'    : 'Hukseflux IR20 pyrgeometer',
                                                'methods'       : 'hemispheric longwave radiation',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})

    lev2_atts['up_short_hemisp']      .update({ 'long_name'     : 'net upward shortwave flux',
                                                'cf_name'       : 'surface_net_upward_shortwave_flux',
                                                'instrument'    : 'Hukseflux SR30 pyranometer',
                                                'methods'       : 'hemispheric shortwave radiation',
                                                'height'        : boom_height,
                                                'location'      : boom_location,})
    
    lev2_atts['s_soil']               .update({'long_name'      : 'soil storage',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10, Hukseflux IR20 pyrgeometer, manual measurements',
                                                'methods'       : '',
                                                'height'        : '',
                                                'location'      : 'soil',})

    lev2_atts['c_soil']               .update({'long_name'      : 'soil conductive flux',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10, Hukseflux IR20 pyrgeometer, manual measurements',
                                                'methods'       : '',
                                                'height'        : '',
                                                'location'      : 'soil',})

    lev2_atts['k_eff_soil']           .update({'long_name'      : 'effective soil thermal conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10, manual measurements',
                                                'methods'       : '',
                                                'height'        : '',
                                                'location'      : 'soil',})
    
    lev2_atts['soil_vwc_5cm']         .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_5cm_depth,
                                                'location'      : 'soil',})
    
    lev2_atts['soil_ka_5cm']          .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_5cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_5cm']           .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_5cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_5cm']          .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_5cm_depth,
                                                'location'      : 'soil',})     
    
    lev2_atts['soil_vwc_10cm']        .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_10cm_depth,
                                                'location'      : 'soil',})
 
    lev2_atts['soil_ka_10cm']         .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_10cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_10cm']          .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_10cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_10cm']         .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_10cm_depth,
                                                'location'      : 'soil',})   
    
    lev2_atts['soil_vwc_20cm']        .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_20cm_depth,
                                                'location'      : 'soil',})
 
    lev2_atts['soil_ka_20cm']         .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_20cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_20cm']          .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_20cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_20cm']         .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_20cm_depth,
                                                'location'      : 'soil',})  
    
    lev2_atts['soil_vwc_30cm']        .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_30cm_depth,
                                                'location'      : 'soil',})
 
    lev2_atts['soil_ka_30cm']         .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_30cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_30cm']          .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_30cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_30cm']         .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_30cm_depth,
                                                'location'      : 'soil',})
    
    lev2_atts['soil_vwc_40cm']        .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_40cm_depth,
                                                'location'      : 'soil',})
 
    lev2_atts['soil_ka_40cm']         .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_40cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_40cm']          .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_40cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_40cm']         .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_40cm_depth,
                                                'location'      : 'soil',})  
    
    lev2_atts['soil_vwc_50cm']        .update({'long_name'      : 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : soil_50cm_depth,
                                                'location'      : 'soil',})
 
    lev2_atts['soil_ka_50cm']         .update({'long_name'      : 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_50cm_depth,
                                                'location'      : 'soil',})       

    lev2_atts['soil_t_50cm']          .update({'long_name'      : 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_50cm_depth,
                                                'location'      : 'soil',})     
  
    lev2_atts['soil_ec_50cm']         .update({'long_name'      : 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : soil_50cm_depth,
                                                'location'      : 'soil',})  

                                            
    return lev2_atts, list(lev2_atts.keys()).copy()


def define_qc_variables(include_turb=False):

    qc_atts = OrderedDict()

    qc_atts['zenith_true_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['azimuth_qc']                 = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['sr50_dist_qc']               = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['snow_depth_qc']              = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['atmos_pressure_qc']          = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['temp_qc']                    = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['rh_qc']                      = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['dew_point_qc']               = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['mixing_ratio_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['vapor_pressure_qc']          = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['rhi_qc']                     = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['brightness_temp_surface_qc'] = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['skin_temp_surface_qc']       = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['subsurface_heat_flux_A_qc']  = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['subsurface_heat_flux_B_qc']  = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['wspd_u_mean_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['wspd_v_mean_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['wspd_w_mean_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['wspd_vec_mean_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['wdir_vec_mean_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['h2o_licor_qc']               = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['co2_licor_qc']               = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['down_long_hemisp_qc']        = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['down_short_hemisp_qc']       = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['up_long_hemisp_qc']          = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['up_short_hemisp_qc']         = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['s_soil_qc']                  = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['c_soil_qc']                  = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['k_eff_soil_qc']              = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_vwc_5cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_5cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_5cm_qc']              = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_5cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_vwc_10cm_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_10cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_10cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_10cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_vwc_20cm_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_20cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_20cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_20cm_qc']            = {'long_name' :'QC flag integer indicating data quality'} 
    qc_atts['soil_vwc_30cm_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_30cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_30cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_30cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_vwc_40cm_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_40cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_40cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_40cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_vwc_50cm_qc']           = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ka_50cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_t_50cm_qc']             = {'long_name' :'QC flag integer indicating data quality'}
    qc_atts['soil_ec_50cm_qc']            = {'long_name' :'QC flag integer indicating data quality'}

    #qc_atts['wind_sector_qc_info']     = {'long_name' : 'QC flag integer indicating wind sector interference conditions'}    
    #qc_atts['wind_sector_qc_info'].update({'comment': 'See global attributes for wind qc specifics.'})

    qc_atts['zenith_true_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['azimuth_qc']                 .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['sr50_dist_qc']               .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['snow_depth_qc']              .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['atmos_pressure_qc']          .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['temp_qc']                    .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['rh_qc']                      .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['dew_point_qc']               .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['mixing_ratio_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['vapor_pressure_qc']          .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['rhi_qc']                     .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['brightness_temp_surface_qc'] .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['skin_temp_surface_qc']       .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['subsurface_heat_flux_A_qc']  .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['subsurface_heat_flux_B_qc']  .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['wspd_u_mean_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['wspd_v_mean_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['wspd_w_mean_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['wspd_vec_mean_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['wdir_vec_mean_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['h2o_licor_qc']               .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['co2_licor_qc']               .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['down_long_hemisp_qc']        .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['down_short_hemisp_qc']       .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['up_long_hemisp_qc']          .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['up_short_hemisp_qc']         .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_vwc_5cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_5cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_5cm_qc']              .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_5cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_vwc_10cm_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_10cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_10cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_10cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_vwc_20cm_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_20cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_20cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_20cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_vwc_30cm_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_30cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_30cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_30cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_vwc_40cm_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_40cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_40cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_40cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_vwc_50cm_qc']           .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_ka_50cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})   
    qc_atts['soil_t_50cm_qc']             .update({'comment': 'See global attributes for qc flag definitions.'})
    qc_atts['soil_ec_50cm_qc']            .update({'comment': 'See global attributes for qc flag definitions.'})

    if include_turb:

        qc_atts['turbulence_qc'] = {'long_name' : 'QC flag integer indicating data quality for all turbulence parameters'}    
        qc_atts['turbulence_qc'].update({'comment': 'See global attributes for qc flag definitions.'})

        qc_atts['bulk_qc']   = {'long_name' : 'QC flag integer indicating data quality for all bulk turbulence parameters'}


    return qc_atts, list(qc_atts.keys()).copy() 

def define_10hz_variables():

    atts_10hz = OrderedDict()
    
    # units defined here, other properties defined in 'update' call below
    atts_10hz['metek_u']           = {'units' : 'm/s'     }
    atts_10hz['metek_v']           = {'units' : 'm/s'     }
    atts_10hz['metek_w']           = {'units' : 'm/s'     }
    atts_10hz['metek_T']           = {'units' : 'C'       }

    atts_10hz['licor_co2']            = {'units' : 'mmol/m3' }
    atts_10hz['licor_h2o']            = {'units' : 'mmol/m3' }

    atts_10hz ['metek_u']  .update({'long_name'  : 'wind velocity in u',
                                    'cf_name'    : 'eastward_wind',
                                    'methods'    : 'u defined positive west-to-east',
                                    'instrument' : 'Metek uSonic-Cage MP sonic anemometer',
                                    'methods'    : 'sonic anemometer; data reported at 20 Hz; TCP/IP protocol', 
                                    'height'     : metek_height,
                                    'location'   : mast_location,})

    atts_10hz['metek_v']   .update({'long_name'  : 'wind velocity in v',
                                    'cf_name'    : 'northward_wind',
                                    'methods'    : 'v defined positive south-to-north',
                                    'instrument' : 'Metek uSonic-Cage MP sonic anemometer',
                                    'methods'    : 'sonic anemometer; data reported at 20 Hz; TCP/IP protocol',
                                    'height'     : metek_height,
                                    'location'   : mast_location,})

    atts_10hz['metek_w']   .update({'long_name'  : 'wind velocity in w',
                                    'methods'    : 'w defined positive up',
                                    'instrument' : 'Metek uSonic-Cage MP sonic anemometer',
                                    'methods'    : 'sonic anemometer; data reported at 20 Hz; TCP/IP protocol',
                                    'height'     : metek_height,
                                    'location'   : mast_location,})

    atts_10hz['metek_T']   .update({'long_name'  : 'acoustic temperature',
                                    'instrument' : 'Metek uSonic-Cage MP sonic anemometer',
                                    'methods'    : 'sonic anemometer; data reported at 20 Hz; TCP/IP protocol',
                                    'height'     : metek_height,
                                    'location'   : mast_location,})

    atts_10hz['licor_co2'] .update({'long_name'  : 'CO2 gas density',
                                    'instrument' : 'Licor 7500-DS',
                                    'methods'    : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol',
                                    'height'     : licor_height,
                                    'location'   : mast_location,})

    atts_10hz['licor_h2o'] .update({'long_name'  : 'water vapor density',
                                    'instrument' : 'Licor 7500-DS',
                                    'methods'    : 'open-path optical gas analyzer, data reported at 20 Hz; TCP/IP protocol',
                                    'height'     : licor_height,
                                    'location'   : mast_location,})

    return atts_10hz, list(atts_10hz.keys()).copy() 


def define_level1_soil():

    lev1_soil_atts = OrderedDict()

    # units defined here, other properties defined in 'update' call below
    lev1_soil_atts['TIMESTAMP']            = {'units' : 'TS'}
    lev1_soil_atts['VWC_5cm']              = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_5cm']               = {'units' : 'unitless'}
    lev1_soil_atts['T_5cm']                = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_5cm']           = {'units' : 'dS/m'}
    lev1_soil_atts['VWC_10cm']             = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_10cm']              = {'units' : 'unitless'}
    lev1_soil_atts['T_10cm']               = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_10cm']          = {'units' : 'dS/m'}
    lev1_soil_atts['VWC_20cm']             = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_20cm']              = {'units' : 'unitless'}
    lev1_soil_atts['T_20cm']               = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_20cm']          = {'units' : 'dS/m'}
    lev1_soil_atts['VWC_30cm']             = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_30cm']              = {'units' : 'unitless'}
    lev1_soil_atts['T_30cm']               = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_30cm']          = {'units' : 'dS/m'}
    lev1_soil_atts['VWC_40cm']             = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_40cm']              = {'units' : 'unitless'}
    lev1_soil_atts['T_40cm']               = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_40cm']          = {'units' : 'dS/m'}
    lev1_soil_atts['VWC_50cm']             = {'units' : 'm3/m3'}
    lev1_soil_atts['Ka_50cm']              = {'units' : 'unitless'}
    lev1_soil_atts['T_50cm']               = {'units' : 'deg'}
    lev1_soil_atts['BulkEC_50cm']          = {'units' : 'dS/m'}
        
    lev1_soil_atts['VWC_5cm']               .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-5 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_5cm']                .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-5 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_5cm']                 .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-5 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_5cm']            .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-5 cm',
                                                'location'      : 'soil',})     
    
    lev1_soil_atts['VWC_10cm']              .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-10 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_10cm']               .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-10 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_10cm']                .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-10 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_10cm']           .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-10 cm',
                                                'location'      : 'soil',})   
    
    lev1_soil_atts['VWC_20cm']              .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-20 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_20cm']               .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-20 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_20cm']                .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-20 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_20cm']           .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-20 cm',
                                                'location'      : 'soil',})  
    
    lev1_soil_atts['VWC_30cm']              .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-30 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_30cm']               .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-30 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_30cm']                .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-30 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_30cm']           .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-30 cm',
                                                'location'      : 'soil',})
    
    lev1_soil_atts['VWC_40cm']              .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-40 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_40cm']               .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-40 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_40cm']                .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-40 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_40cm']           .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-40 cm',
                                                'location'      : 'soil',})  
    
    lev1_soil_atts['VWC_50cm']              .update({'long_name': 'volume water content',
                                                'cf_name'       : 'volume fraction of condenced water in soil',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR), insensitive to frozen soils',
                                                'height'        : '-50 cm',
                                                'location'      : 'soil',})
 
    lev1_soil_atts['Ka_50cm']               .update({'long_name': 'relative permittivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-50 cm',
                                                'location'      : 'soil',})       

    lev1_soil_atts['T_50cm']                .update({'long_name': 'temperature',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-50 cm',
                                                'location'      : 'soil',})     
  
    lev1_soil_atts['BulkEC_50cm']           .update({'long_name': 'bulk electrical conductivity',
                                                'cf_name'       : '',
                                                'instrument'    : 'SoilVUE10',
                                                'methods'       : 'time-domain reflectometry (TDR)',
                                                'height'        : '-50 cm',
                                                'location'      : 'soil',})  

    return lev1_soil_atts, list(lev1_soil_atts.keys()).copy()