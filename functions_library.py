# ############################################################################################
# define here the functions that are used across the different pieces of code. 
# we want to be sure something doesnt change in one module and not another
#
# The following functions are defined here (in this order):
#
#       def despike(spikey_panda, thresh, filterlen):                                                        
#       def calculate_metek_ws_wd(data_dates, u, v, hdg_data): # tower heading data is 1s, metek data is 1m  
#       def calc_humidity_ptu300(RHw, temp, press, Td):                                                      
#       def tilt_rotation(ct_phi, ct_theta, ct_psi, ct_up, ct_vp, ct_wp):                                    
#       def decode_licor_diag(raw_diag):                                                                     
#       def get_ct(licor_db):                                                                                
#       def get_dt(licor_db):                                                                                
#       def get_pll(licor_db):                                                                               
#       def take_average(array_like_thing, **kwargs):                                                        
#       def num_missing(series):                                                                             
#       def perc_missing(series):                                                                            
#       def column_is_ints(ser):                                                                             
#       def warn(string):                                                                                    
#       def fatal(string):                                                                                   
#       def grachev_fluxcapacitor(z_level_nominal, z_level_n, sonic_dir, metek, licor, clasp, verbose=False):
#
# ############################################################################################
import pandas as pd
import numpy  as np
import scipy  as sp


from datetime import datetime, timedelta
from scipy    import signal

global nan; nan = np.NaN

# despiker
def despike(spikey_panda, thresh, filterlen, medfill):
    # outlier detection from a running median !!!! Replaces outliers with that median !!!!
    tmp                    = spikey_panda.rolling(window=filterlen, center=True, min_periods=1).median()
    spikes_i               = (np.abs(spikey_panda - tmp)) > thresh
    if medfill == 'yes': # fill with median
        spikey_panda[spikes_i] = tmp
    elif medfill == 'no': # fill with nan
        spikey_panda[spikes_i] = np.nan
    else: # just return bad indices
        spikey_panda = spikes_i
    return spikey_panda

# calculate humidity variables following Vaisala
def calc_humidity_ptu300(RHw, temp, press, Td):

    # Calculations based on Appendix B of the PTU/HMT manual to be mathematically consistent with the
    # derivations in the on onboard electronics. Checked against Ola's code and found acceptable
    # agreement (<0.1% in MR). RHi calculation is then made following Hyland & Wexler (1983), which
    # yields slightly higher (<1%) compared a different method of Ola's

    # calculate saturation vapor pressure (Pws) using two equations sets, Wexler (1976) eq 5 & coefficients
    c0    = 0.4931358
    c1    = -0.46094296*1e-2
    c2    = 0.13746454*1e-4
    c3    = -0.12743214*1e-7
    omega = temp - ( c0*temp**0 + c1*temp**1 + c2*temp**2 + c3*temp**3 )

    # eq 6 & coefficients
    bm1 = -0.58002206*1e4
    b0  = 0.13914993*1e1
    b1  = -0.48640239*1e-1
    b2  = 0.41764768*1e-4
    b3  = -0.14452093*1e-7
    b4  = 6.5459673
    Pws = np.exp( ( bm1*omega**-1 + b0*omega**0 + b1*omega**1 + b2*omega**2 + b3*omega**3 ) + b4*np.log(omega) ) # [Pa]

    Pw = RHw*Pws/100 # # actual vapor pressure (Pw), eq. 7, [Pa]

    x = 1000*0.622*Pw/((press*100)-Pw) # mixing ratio by weight (eq 2), [g/kg]

    # if we no dewpoint available (WXT!) then calculate it, else no need to worry about it
    if Td == -1:   # dewpoint (frostpoint), we are assuming T ambient < 0 C, which corresponds to these coefs:
        A = 6.1134
        m = 9.7911
        Tn = 273.47
        Td = Tn / ((m/np.log10((Pw/100)/A)) - 1) # [C] (temperature, not depression!)

    # else: do nothing if arg 4 is any other value and input flag will be returned.
    a = 216.68*(Pw/temp) # # absolute humidity, eq 3, [g/m3]

    h = (temp-273.15)*(1.01+0.00189*x)+2.5*x # ...and enthalpy, eq 4, [kJ/kg]

    # RHi, the saturation vapor pressure over ice, then finally RHI, Hyland & Wexler (1983)
    c0 = -5.6745359*1e3     # coefficients
    c1 = 6.3925247
    c2 = -9.6778430*1e-3
    c3 = 6.2215701*1e-7
    c4 = 2.0747825*1e-9
    c5 = -9.4840240*1e-13
    D  = 4.1635019

    # calculate
    term = (c0*temp**(-1)) + (c1*temp**(0)) + (c2*temp**1) + (c3*temp**2) + (c4*temp**3)+(c5*temp**4)

    # calculate saturation vapor pressure over ice
    Psi = np.exp(term + (D*np.log(temp)))  # Pa

    # convert to rhi
    rhi = 100*(RHw*0.01*Pws)/Psi

    return Td, h, a, x, Pw, Pws, rhi

def calculate_initial_angle_wgs84(latA,lonA,latB,lonB):
    
    # a little more accurate than calculate_initial_angle, which assumed a great circle. here we match gps datum wgs84
    from pyproj import Geod  

    g = Geod(ellps='WGS84')
    az12,az21,dist = g.inv(lonA,latA,lonB,latB)
    compass_bearing = az12
    return compass_bearing
  
def distance_wgs84(latA,lonA,latB,lonB):
    
    # a little more accurate than distance, which assumed a great circle. here we match gps datum wgs84
    from pyproj import Geod  

    g = Geod(ellps='WGS84')
    az12,az21,dist = g.inv(lonA,latA,lonB,latB)
    d = dist
    return d
      
def calculate_initial_angle(latA,lonA, latB, lonB):

    # Function provided by Martin Radenz, TROPOS
    
    # Calculates the bearing between two points.
    # The formulae used is the following:
    #     ? = atan2(sin(?long).cos(lat2),
    #               cos(lat1).sin(lat2) ? sin(lat1).cos(lat2).cos(?long))

    # source: https://gist.github.com/jeromer/2005586

    # initial_bearing = math.degrees(initial_bearing)
    # compass_bearing = (initial_bearing + 360) % 360

    # :Parameters:
    #   - `pointA: The tuple representing the latitude/longitude for the
    #     first point. Latitude and longitude must be in decimal degrees
    #   - `pointB: The tuple representing the latitude/longitude for the
    #     second point. Latitude and longitude must be in decimal degrees
    # :Returns:
    #   The bearing in degrees
    # :Returns Type:
    #   float

#     if (type(pointA) != tuple) or (type(pointB) != tuple):
#         raise TypeError("Only tuples are supported as arguments")

    lat1 = np.radians(latA)
    lat2 = np.radians(latB)

    diffLong = np.radians(lonB - lonA)

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
            * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180∞ to + 180∞ which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below

    #print((math.degrees(initial_bearing) + 360) % 360)
    initial_bearing = initial_bearing/np.pi*180
    
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def distance(lat1, lon1, lat2, lon2):
    
    # Function provided by Martin Radenz, TROPOS
        
    #     lat1, lon1 = origin
    #     lat2, lon2 = destination
    radius = 6361 # km

    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1))         * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d


def tilt_rotation(ct_phi, ct_theta, ct_psi, ct_up, ct_vp, ct_wp):

    # This subroutine rotates a vector from one cartesian basis to another, based upon the
    # three Euler angles, defined as rotations around the reference axes, xyz.

    # Rotates the sonic winds from body coordinates to earth coordinates.  This is the tild
    # rotation that corrects for heading and the tilt of the sonic reltive to plum.

    # y,x,z in -> u,v,w out

    # This differs from the double rotation into the plane of the wind streamline that is
    # needed for the flux calculations and is performed in the grachev_fluxcapacitor. The
    # output from this rotation is the best estimate of the actual wind direction, having
    # corrected for both heading and contributions to the horizontal wind by z

    # Adapted from coord_trans.m (6/27/96) from Chris Fairall's group by C. Cox 2/22/20
    #
    #  phi   = about inclinometer y/u axis (anti-clockwise about Metek north-south) (roll)
    #  theta = about inclinometer x/v axis (anti-clockwise about Metek east-west) (pitch)
    #  psi   = about z/w axis (yaw, "heading")

    # calculations are in radians, but inputs are in degrees
    ct_phi   = np.radians(ct_phi)
    ct_theta = np.radians(ct_theta)
    ct_psi   = np.radians(ct_psi)

    ct_u = ct_up*np.cos(ct_theta)*np.cos(ct_psi) + ct_vp*(np.sin(ct_phi)*np.sin(ct_theta)*np.cos(ct_psi)-np.cos(ct_phi)*np.sin(ct_psi)) + ct_wp*(np.cos(ct_phi)*np.sin(ct_theta)*np.cos(ct_psi)+np.sin(ct_phi)*np.sin(ct_psi))

    ct_v = ct_up*np.cos(ct_theta)*np.sin(ct_psi) + ct_vp*(np.sin(ct_phi)*np.sin(ct_theta)*np.sin(ct_psi)+np.cos(ct_phi)*np.cos(ct_psi)) + ct_wp*(np.cos(ct_phi)*np.sin(ct_theta)*np.sin(ct_psi)-np.sin(ct_phi)*np.cos(ct_psi))

    ct_w = ct_up*(-np.sin(ct_theta)) + ct_vp*(np.cos(ct_theta)*np.sin(ct_phi)) + ct_wp*(np.cos(ct_theta)*np.cos(ct_phi))

    return ct_u, ct_v, ct_w

# inst_prefix is just a name that you prepended to the standard fast vars. can be an empty string
def fix_high_frequency(fast_data, inst_prefix=''):

    inst = inst_prefix # shorthand

    # corrections to the high frequency component of the turbulence spectrum... the metek
    # sonics used seem to have correlated cross talk between T and w that results in biased
    # flux values with a dependency on frequency...
    #
    # this correction fixes that and is documented in the data paper

    # these are copies of turb_data['fs'+suffix_list[i_inst]], from the fluxcapicitor, 
    # but that hasn't been created yet because we aren't there
    fs = np.array([0,0.0012,0.0024,0.0037,0.0049,0.0061,0.0079,0.0104,0.0128,0.0153,0.0183,0.0220,0.0256,
                   0.0299,0.0348,0.0397,0.0452,0.0519,0.0592,0.0671,0.0763,0.0867,0.0977,0.1099,0.1239,
                   0.1392,0.1556,0.1740,0.1947,0.2179,0.2435,0.2716,0.3027,0.3369,0.3748,0.4169,0.4633,
                   0.5145,0.5713,0.6342,0.7037,0.7806,0.8655,0.9595,1.0638,1.1792,1.3062,1.4465,1.6022,
                   1.7743,1.9647,2.1753,2.4078,2.6648,2.9486,3.2623,3.6090,3.9923,4.4165,4.8193])

    # sf = scale factors; ie, af(x)) from Eq.(5).
    sf = np.concatenate([np.tile(0,47),
                         np.array([0.0001,0.0030,0.0069,0.0115,0.0174,0.0246,0.0338,
                                   0.0455,0.0597,0.0767,0.0958,0.1134,0.1229])])

    # Eq. (5) from data paper?
    Num   = np.int(np.ceil(np.log2(np.size(fast_data[inst+'w']))))
    freqw = np.fft.fft(fast_data[inst+'w'].fillna(fast_data[inst+'w'].median()),2**Num)
    freqf = (10/2**Num)*np.arange(0,2**(Num-1)) # frequencies of the fft. 10 is 10 Hz sampling freq. can this be softcoded?

    # sf curve is coarsely sampled, so interpolate to freqf
    sfinterp = np.interp(freqf,np.transpose(fs),sf)  
    goback   = np.real(np.fft.ifft(freqw*np.concatenate([sfinterp,np.flipud(sfinterp)]),2**Num))  # ifft of a(f(x))*yw


    # subtract off the noise
    fast_data[inst+'T'] = fast_data[inst+'T']-goback[:np.size(fast_data[inst+'w'])]  

    return fast_data


def decode_licor_diag(raw_diag):

    # speed things up so we dont wait forever
    bin_v     = np.vectorize(bin)
    get_ct_v  = np.vectorize(get_ct)
    get_dt_v  = np.vectorize(get_dt)
    get_pll_v = np.vectorize(get_pll)

    # licor diagnostics are encoded in the binary of an integer reported by the sensor. the coding is described
    # in Licor Technical Document, 7200_TechTip_Diagnostic_Values_TTP29 and unpacked here.
    licor_diag     = np.int16(raw_diag)
    pll            = licor_diag*np.nan
    detector_temp  = licor_diag*np.nan
    chopper_temp   = licor_diag*np.nan
    # __"why the %$@$)* can't this be vectorized? This loop is slow"__
    # well, chris, you asked for vectorization.. now you got it! now to get rid of the list comprehension
    non_nan_inds  = ~np.isnan(raw_diag) 
    #if not np.isnan(raw_diag).all():
    if non_nan_inds.any():

        licor_diag_bin = bin_v(licor_diag[non_nan_inds])

        # asfs radio data doesn't contain good licor_diags, so this fails, treat as if we had nans
        try: chopper_temp_bin = licor_diag_bin[:][2]
        except:
            print("!!! no licor diags for today !!!")
            pass

        # try to use vectorized functions?
        # chopper_temp[non_nan_inds]  = get_ct_v(licor_diag_bin)
        # detector_temp[non_nan_inds] = get_dt_v(licor_diag_bin)
        # pll[non_nan_inds]           = get_pll_v(licor_diag_bin)
        
        # or just use list comprehension? vectorize is doing weird things, and not much faster... maybe this is better
        chopper_temp[non_nan_inds]  = [np.int(x[2]) if len(x)==10 and x[2]!='b' else np.nan for x in licor_diag_bin]
        detector_temp[non_nan_inds] = [np.int(x[3]) if len(x)==10 and x[2]!='b' else np.nan for x in licor_diag_bin]
        pll[non_nan_inds]           = [np.int(x[4]) if len(x)==10 and x[2]!='b' else np.nan for x in licor_diag_bin]

    return pll, detector_temp, chopper_temp

# these functions are for vectorization by numpy to 
def get_ct(licor_db):
    if len(licor_db)>4 and licor_db[2]!='b':
        if licor_db[2] != np.nan:
            return np.int(licor_db[2])
    else: return np.nan
def get_dt(licor_db):
    if len(licor_db)>4 and licor_db[2]!='b': 
        if licor_db[3] != np.nan:
            return np.int(licor_db[3])
    else: return np.nan
def get_pll(licor_db):
    if len(licor_db)>4 and licor_db[2]!='b':
        if licor_db[4] != np.nan:
            return np.int(licor_db[4])
    else: return np.nan

# this is the function that averages for the 1m and 10m averages
# perc_allowed_missing defines how many data points are required to be non-nan before returning nan
# i.e. if 10 out of 100 data points are non-nan and you specify 80, this returns nan
#
# can be called like:     DataFrame.apply(take_average, perc_allowed_missing=80)
def take_average(array_like_thing, **kwargs):

    # this is for exceptions where you pass this function dates and such
    UFuncTypeError = np.core._exceptions.UFuncTypeError
    perc_allowed_missing = kwargs.get('perc_allowed_missing')
    if perc_allowed_missing is None:
        perc_allowed_missing = 50.0
    
    if array_like_thing.size == 0:
        return np.nan
    perc_miss = np.round((np.count_nonzero(np.isnan(array_like_thing))/float(array_like_thing.size))*100.0, decimals=4)
    if perc_miss > perc_allowed_missing:
        return np.nan
    else:
        try: mean_val = np.nanmean(array_like_thing)
        except UFuncTypeError as e: 
            mean_val = np.nan #  this exception should only happen when this function is used on a pandas array
                              #  that contains dates or other things where averages are hard to define
        return mean_val

def take_vector_average(array_like_thing, **kwargs):

    # this is for exceptions where you pass this function dates and such
    UFuncTypeError = np.core._exceptions.UFuncTypeError
    perc_allowed_missing = kwargs.get('perc_allowed_missing')
    if perc_allowed_missing is None:
        perc_allowed_missing = 50.0
    
    if array_like_thing.size == 0:
        return np.nan
    perc_miss = np.round((np.count_nonzero(np.isnan(array_like_thing))/float(array_like_thing.size))*100.0, decimals=4)
    if perc_miss > perc_allowed_missing:
        return np.nan
    else:
        try: mean_val = np.nanmean(array_like_thing)
        except UFuncTypeError as e: 
            mean_val = np.nan #  this exception should only happen when this function is used on a pandas array
                              #  that contains dates or other things where averages are hard to define
        return mean_val


# functions to make grepping lines easier, differentiating between normal output, warnings, and fatal errors
def warn(string):
    max_line = len(max(string.splitlines(), key=len))
    print('')
    print("!! Warning: {} !!".format("!"*(max_line)))
    for line in string.splitlines():
        print("!! Warning: {} {}!! ".format(line," "*(max_line-len(line))))
    print("!! Warning: {} !!".format("!"*(max_line)))
    print('')

def fatal(string):
    max_line = len(max(string.splitlines(), key=len))
    print('')
    print("!! FATAL {} !!".format("!"*(max_line)))
    for line in string.splitlines():
        print("!! FATAL {} {}!! ".format(line," "*(max_line-len(line))))
    center_off = int((max_line-48)/2.)
    if center_off+center_off != (max_line-len(line)):
        print("!! FATAL {} I'm sorry, but this forces an exit... goodbye! {} !!".format(" "*center_off," "*(center_off)))
    else:
        print("!! FATAL {} I'm sorry, but this forces an exit... goodbye! {} !!".format(" "*center_off," "*center_off))
    print("!! FATAL {} !!".format("!"*(max_line)))
    exit()

def num_missing(series):
    return np.count_nonzero(series==np.NaN)

def perc_missing(series):
    if series.size == 0: return 100.0
    return np.round((np.count_nonzero(np.isnan(series))/float(series.size))*100.0, decimals=4)

# checks a column/series in a dataframe to see if it can be stored as ints 
def column_is_ints(ser): 

    if ser.dtype == object:
        warn("your pandas series for {} contains objects, that shouldn't happen".format(ser.name))
        return False

    elif ser.empty or ser.isnull().all():
        return False
    else:
        mx = ser.max()
        mn = ser.min()

        ser_comp = ser.fillna(mn-1)  
        ser_zero = ser.fillna(0)

        # test if column can be converted to an integer
        try: asint  = ser_zero.astype(np.int32)
        except Exception as e:
            return False
        result = (ser_comp - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:

            # could probably support compression to smaller dtypes but not yet,
            # for now everything is int32... because simple

            # if mn >= 0:
            #     if mx < 255:
            #         return np.uint8
            #     elif mx < 65535:
            #         return np.uint16
            #     elif mx < 4294967295:
            #         return np.uint32
            #     else:
            #         print("our netcdf files dont support int64")
            # else:
            #     if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
            #         return np.int8
            #     elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
            #         return np.int16
            #     elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
            #         return np.int32
            #     elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
            #         print("our netcdf files dont support int64")

            return True
        else:
            return False

# despik.m
def despik(uraw):

    uz = np.array(uraw)
    on=np.array(range(0, len(uz)))
    uz=np.column_stack([on,uz])
    
    npt=np.size(uz,0)
    
    uu2=uz[uz[:,1].argsort(),]
    uu=uu2[0:npt,1]
    mp=np.floor(npt/2)
    mu=uu[int(mp)]
    sp=np.floor(0.84*npt)
    sm=np.floor(0.16*npt)
    sig=(uu[int(sp)]-uu[int(sm)])/2
    dsig=max(4*sig,0.5)
    im=1
    while abs(mu-uu[im])>dsig:
       im=im+1
    
    ip=npt-1
    while abs(uu[ip]-mu)>dsig:
       ip=ip-1
    
    pct=(im+npt-ip)/npt*100
    uu2[0:im,1]=mu
    uu2[ip:npt,1]=mu
    uy=uu2[uu2[:,0].argsort(),]
    uu=uy[0:npt,1]
    
    return uu

# maybe this goes in a different file?
def grachev_fluxcapacitor(z_level_n, metek, licor, h2ounit, co2unit, pr, temp, mr, le_flag, b0 ,b1 ,b2, rotflag,
                          verbose=False, integration_window=None):
    
    # define the verbose print option
    v_print      = print if verbose else lambda *a, **k: None
    verboseprint = v_print
    nan          = np.NaN  # make using nans look better, in the eyes of the bee-holder 

    # some setup
    samp_freq = 10             # sonic sampling rate in Hz
    npos      = 1800*samp_freq # highest possible number of data points (36000)
    sfreq     = 1/samp_freq
    nx        = samp_freq*(1800-1)

    if np.isnan(mr): mr = 0 # if we don't have a value, just assume dry air. it's a formality anyhow.
    if np.isnan(temp): temp = metek['T'].mean()
    if np.isnan(pr): pr = 1013 # Andrey's nominal value
        
    #++++++++++++++++++++++++ Important constants and equations ++++++++++++++++++++++++++++++++
    # !!! temperature in deg C
    tdk   = 273.15                                                      # C->K
    Rd    = 287.1                                                       # [J/(kg K)] universal gas constant
    Rv    = 461                                                         # [J/(kg K)] gas constant, water vapor
    pp_wv = mr / (mr + 0.622) * pr*100                                  # [Pa] partial prssure from water vapor             
    rho_d = ((pr*100)-pp_wv)/(Rd*(temp+tdk))                            # [kg/m3] density of dry air
    rho_v = pp_wv/(Rv*(temp+tdk))                                       # [kg/m3] desity of water vapor 
    rho   = rho_d + rho_v                                               # [kg/m3] total density of the mosit air #rho = pr*100/(Rgas*(temp+tdk)*(1+0.61e-3*mr)) 
    sigma = rho_v/rho_d                                                 # [unitless] rho_v/rho_dry                                  
    cp    = 1005.6+0.017211*temp+0.000392*temp**2                       # [J/(kg K)] isobaric specific heat of air (median), (from Ed Andreas)
    visa  = 1.326e-5*(1+6.542e-3*temp+8.301e-6*temp**2-4.84e-9*temp**3) # [m^2/s] molecular viscosity from TOGA CORE bulk algorithm
    kt    = (0.02411*(1+0.003309*temp-1.441e-6*temp**2))/(rho*cp)       # [m^2/s] coefficient of molecular thermal diffusivity (from Ed Andreas)
    M_h2o = 18.01528/1000                                               # [kg/mol] molar mass, water 
    M_co2 = 44.01/1000                                                  # [kg/mol] molar mass, co2
    
    if le_flag == 1: # snow cover
        Le    = (2.501-.00237*temp)*1e6 + 333.55*1000 # [J/kg] latent heat of vaporization 
    elif le_flag == 0: # snow free
        Le    = (2.501-.00237*temp)*1e6 # [J/kg] latent heat of vaporization                                    

    # this should subset to 30 min and loop. get to it later.
    U   = metek['u']
    V   = metek['v']
    W   = metek['w']
    T   = metek['T']
    
    # get the gases into the right units
    if 'g/m3' in h2ounit:
        Q = licor['licor_h2o']/1000 # kg/m3
    elif 'mmol/m3' in h2ounit:
        Q = licor['licor_h2o']/1000 * M_h2o # mmol/m3 -> mol/m^3 -> kg/m3
        
    if 'mg/m3' in co2unit:
         C = licor['licor_co2']      # mg/m3
    elif 'mmol/m3' in co2unit:
         C = (licor['licor_co2']/1000 * M_co2)*1e6 # mmol/m3 -> mol/m_3 -> kg/m3 -> mg/m3
    
    #H = H * Rgas*(T+tdk)/(pr*100) # gas density to mixing ratio
    Q = Q / (Q + rho) # to specfic humidity

    npt = len(U)

    # powers of two for the underlying DFT to be integrated
    # for half hour data we expect to be here, a 27.3 min flux, the closest po2 for the FFT to 30 min.
    fail = False
    if integration_window == 3:   nf = 2**11
    elif integration_window == 5:   nf = 2**12 
    elif integration_window == 10:   nf = 2**13 # 2^13 = 8192 points  (or = 8192/10 sec  = 13.65333 min for samp_freq = 10 Hz)
    elif integration_window == 30: nf = 2**14 # 2^14 = 16384 points (or = 16384/10 sec = 27.30667 min for samp_freq = 10 Hz)
    elif integration_window == 60: nf = 2**15 # 2^15 = 32768 points (or = 32768/10 sec = 54.61333 min for samp_freq = 10 Hz)
    else:
         print(f"!!! YOU'VE PASSED AN INVALID WINDOW TO THE FLUXCAPACITOR ::{integration_window}"); fail = True

    # Sanity check: Reject series if it is too short, less than 2^13 = 8192 points = 13.6 min @ 10 Hz
    if npt < nf:
        print(f"!!! YOU'VE REQUESTED A WINDOW OF {integration_window} BUT YOUR DATA WAS TOO SHORT: {npt}"); fail = True

    # Goodness, there are going to be a lot of things to save. Lets package it up.
    turbulence_data = pd.DataFrame(columns=[\
    'Hs',            # sensible heat flux (W/m2) - Based on the sonic temperature!!!
    'Hl',            # Latent heat flux
    'CO2_flux',      # co2 flux in mg m-2 s-1
    'CO2_flux_Wm2',  # co2 flux (W/m2) in mg m-2 s-1, Webb correction applied
    'Cd',            # Drag coefficient, Cd
    'ustar',         # the friction velocity based only on the downstream, uw,  stress components (m/s)
    'Tstar',         # the temperature scale
    'zeta_level_n',  # Monin-Obukhov stability parameter, z/L
    'WU_csp',        # wu-covariance based on the wu-cospectra integration
    'WV_csp',        # wv-covariance based on the wv-cospectra integration
    'UV_csp',        # uv-covariance based on the uv-cospectra integration
    'WT_csp',        # wt-covariance, vertical flux of the sonic temperature  [deg m/s]
    'UT_csp',        # ut-covariance, horizontal flux of the sonic temperature (along-wind component) [deg m/s]
    'VT_csp',        # vt-covariance, horizontal flux of the sonic temperature (cross-wind component) [deg m/s]
    'Wq_csp',        # wt-covariance, vertical flux of vapor  [kg/m3 m/s]
    'Uq_csp',        # ut-covariance, horizontal flux of vapor (along-wind component) [kg/m3 m/s]
    'Vq_csp',        # vt-covariance, horizontal flux of co2 (cross-wind component) [kg/m3 m/s]
    'Wc_csp',        # wt-covariance, vertical flux of co2  [mg m-2 s-1 m/s]
    'Uc_csp',        # ut-covariance, horizontal flux of co2 (along-wind component) [mg m-2 s-1 deg m/s]
    'Vc_csp',        # vt-covariance, horizontal flux of vapor (cross-wind component) [mg m-2 s-1 m/s]
    'phi_U',         # MO universal functions for the standard deviations
    'phi_V',         # MO universal functions for the standard deviations
    'phi_W',         # MO universal functions for the standard deviations
    'phi_T',         # MO universal functions for the standard deviations
    'phi_UT',        # MO universal function for the horizontal heat flux
    'epsilon_U',     # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the longitudinal velocity component in the inertial subrange [m^2/s^3]:
    'epsilon_V',     # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the lateral velocity component in the inertial subrange [m^2/s^3]:
    'epsilon_W',     # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the vertical velocity component in the inertial subrange [m^2/s^3]:
    'epsilon',       # Dissipation rate of the turbulent kinetic energy = median of the values derived from Su, Sv, & Sw [m^2/s^3]:
    'Phi_epsilon',   # Monin-Obukhov universal function Phi_epsilon based on the median epsilon:
    'nSU',           # inertial subrange slope
    'nSV',           # inertial subrange slope
    'nSW',           # inertial subrange slope
    'nST',           # inertial subrange slope
    'nSq',           # inertial subrange slope
    'nSc',           # inertial subrange slope
    'NT',            # The dissipation (destruction) rate for half the temperature variance [deg^2/s]
    'Phi_NT',        # Monin-Obukhov universal function Phi_Nt
    'sigU',          # QC: standard deviation of the rotated u (streamwise)
    'sigV',          # QC: standard deviation of the rotated v (cross stream)
    'sigW',          # QC: standard deviation of the rotated w (perpendciular to stream)
    'Phix',          # QC: Angle of attack
    'DeltaU',        # QC: Non-Stationarity
    'DeltaV',        # QC: Non-Stationarity
    'DeltaT',        # QC: Non-Stationarity  
    'Deltaq',        # QC: Non-Stationarity 
    'Deltac',        # QC: Non-Stationarity 
    'sUs',           # Variance spectrum
    'sVs',           # Variacne spectrum
    'sWs',           # Variance spectrum
    'sTs',           # Variance spectrum
    'sqs',           # Variance spectrum
    'scs',           # Variance spectrum
    'cWUs',          # Cospectrum
    'cWVs',          # Cospectrum
    'cWTs',          # Cospectrum
    'cUTs',          # Cospectrum
    'cVTs',          # Cospectrum
    'cWqs',          # Cospectrum
    'cUqs',          # Cospectrum
    'cVqs',          # Cospectrum
    'cWcs',          # Cospectrum
    'cUcs',          # Cospectrum
    'cVcs',          # Cospectrum
    'cUVs',          # Cospectrum
    'dfs',
    'fs'])           # Frequency vector

    if fail:
        try: verboseprint(f'!!! no valid data on {metek.index[0]} for sonic at height ({npt=}, {nf=}) {np.str(z_level_n)}')
        except: verboseprint(f'!!! no valid data on {metek} for sonic at height ({npt=}, {nf=}) {np.str(z_level_n)}')

        # give the cols unique names (for netcdf later), give it a row of nans, and kick it back to the main
        turbulence_data.keys    = turbulence_data.keys()#+'_'+z_level_nominal
        turbulence_data.columns = turbulence_data.keys
        turbulence_data         = turbulence_data.append([{turbulence_data.keys[0]: np.nan}])  
        return turbulence_data

    # despike following Fairall et al.
    U[:] = despik(U)
    V[:] = despik(V)
    W[:] = despik(W)
    T[:] = despik(T)
    Q[:] = despik(Q)
    T[:] = despik(T)

    # reject the series of more than 49.9998% of u- or v- or w-wind speed component are nan
    # the -2 is to account for weird oddities of certain subset of weird asfs days where the logging went haywire
    # but we would really prefer to not completely trash the data
    if sum(U.isna()) >= np.floor((npt/2)-2) or sum(V.isna()) >= np.floor((npt/2)-2) \
     or sum(W.isna()) >= np.floor((npt/2)-2) or sum(T.isna()) >= np.floor((npt/2)-2):

        verboseprint('  No valid data for sonic at height (>50% missing)  '+np.str(z_level_n))

        turbulence_data.keys    = turbulence_data.keys()#+'_'+z_level_nominal
        turbulence_data.columns = turbulence_data.keys
        turbulence_data         = turbulence_data.append([{turbulence_data.keys[0]: nan}])       
        return turbulence_data

    # special case for Licor. If the licor has inssufficient data, we still want to run the code for Hs so we will
    # set licor to -9999 so that nan-sensitive operations can complete then return it to nan at the end of the code
    licor_missing = 0 # assume licor data is suffcient
    if sum(Q.isna()) > np.floor((npt/2)-2):
        Q[:] = -9999.
        C[:] = -9999.        
        licor_missing = 1
        
    # Replace missing values (NaN) or inf using mean values:
    # This method is preferable in the case if there are too many NaNs or long gaps
    U[np.isinf(U)] = nan
    V[np.isinf(V)] = nan
    W[np.isinf(W)] = nan
    T[np.isinf(T)] = nan
    Q[np.isinf(Q)] = nan
    C[np.isnan(C)] = nan

    U = U.fillna(U.mean())
    V = V.fillna(V.mean())
    W = W.fillna(W.mean())
    T = T.fillna(T.mean())
    Q = Q.fillna(Q.mean())
    C = C.fillna(C.mean())

    # Computes means for entire time block (1 hr or less)
    um = U.mean()
    vm = V.mean()
    wm = W.mean()
    Tm = T.mean()
    Qm = Q.mean()
    Cm = C.mean()

    # there was one day for asfs30 (10/06/2019) where this was true and so wsp divisions were undefined
    if um == 0 and vm ==0 and wm ==0: #half hour of _zero_ wind? has to be bad data
        turbulence_data.keys    = turbulence_data.keys()#+'_'+z_level_nominal
        turbulence_data.columns = turbulence_data.keys
        turbulence_data         = turbulence_data.append([{turbulence_data.keys[0]: nan}])
        verboseprint('  Bad data for sonic at height {}'.format(np.str(z_level_n)))
        return turbulence_data

    # Rotate!
    if rotflag == 1: # DOUBLE ROTATION
    
        # method used until 14 March 2022 (commented out rot, phi, x, xr lines below)    
        # We used the most common method, which is a double rotation of the anemometer coordinate system, to
        # compute the longitudinal, lateral, and vertical velocity components in real time (Kaimal and Finnigan
        # 1994, Sect. 6.6).
        #
        # As of 14 Mar 2022 we swtiched to a single azimuthal rotation to manage the w bias in the uSonic
        #
        # working in radians in this block
        thet     = np.arctan2(vm,um)
        ss = (um**2+vm**2)**0.5
        phi      = np.arctan2(wm,ss)
        rot      = np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]])
        rot[0,0] = np.cos(phi)*np.cos(thet)
        rot[0,1] = np.cos(phi)*np.sin(thet)
        rot[0,2] = np.sin(phi)
        rot[1,0] = -np.sin(thet)
        rot[1,1] = np.cos(thet)
        rot[1,2] = 0
        rot[2,0] = -1*np.sin(phi)*np.cos(thet)
        rot[2,1] = -1*np.sin(phi)*np.sin(thet)
        rot[2,2] = np.cos(phi)
        x=np.array([U,V,W])
        xr=np.dot(rot,x)
        U = xr[0,:]
        V = xr[1,:]
        W = xr[2,:]
    
    elif rotflag == 2: # PLANAR FIT
        
        thet     = np.arctan2(vm,um)
        ss       = (um**2+vm**2)**0.5
        phi      = np.arctan2(wm,ss)
        p31      = -b1 / np.sqrt(b1**2+b2**2+1)
        p32      = -b2 / np.sqrt(b1**2+b2**2+1)
        p33      = 1 / np.sqrt(b1**2+b2**2+1)
        
        alpha    = np.arcsin(p31)
        beta     = np.arcsin(-p32/np.sqrt(p32**2+p33**2))
        
        alpha    = beta*-1 # Wilczak et al is for left-handed coords. This puts us in right hand for metek
        beta     = alpha*-1 # Wilczak et al is for left-handed coords. This puts us in right hand for metek
        
        sinalpha = np.sin(alpha)
        sinbeta  = np.sin(beta)
        cosbeta  = np.cos(beta)
        cosalpha = np.cos(alpha)

        Cmat     = np.array([[1.,0.,0.],[0.,cosbeta,-sinbeta],[0.,sinbeta,cosbeta]]) 
        Dmat     = np.array([[cosalpha,0.,sinalpha],[0.,1.,0.],[-sinalpha,0.,cosalpha]]) 
        Pmat     = np.matmul(np.transpose(Dmat),np.transpose(Cmat))
        x        = np.array([U,V,W-b0])
        xr       = np.matmul(Pmat,x)
        gmma     = np.arctan(xr[1,:].mean()/xr[0,:].mean())
        Mmat     = np.array([[np.cos(gmma), np.sin(gmma),0.],[-np.sin(gmma),np.cos(gmma),0],[0.,0.,1.]]) 
        xr       = np.matmul(Mmat,xr)
        U = xr[0,:]
        V = xr[1,:]
        W = xr[2,:]
        
    elif rotflag == 3: # SINGLE ROTATE
    
        # just the azimuthal rotation
        thet     = np.arctan2(vm,um)
        ss = (um**2+vm**2)**0.5
        phi      = np.arctan2(wm,ss)
        U = np.array (  U*np.cos(thet) + V*np.sin(thet) )
        V = np.array ( -U*np.sin(thet) + V*np.cos(thet) )  
        W = np.array (  W )

    
    # Mean rotated wind speed components:
    # vrm and wrm can be used for QC purposes (theoretically vrm = 0 and wrm = 0)
    urm = U.mean()
    vrm = V.mean()
    wrm = W.mean()
    # The standard deviations of the rotated components are needed by Ola's QC routines based on SHEBA
    urs = U.std()
    vrs = V.std()
    wrs = W.std()
    
    #
    # Compute the spectra and cospectra
    #

    nf_min = nf/600     # nf in minutes

    # Instantaneous relative direction of the wind speed vector in the rotated coordinate system:
    wdirr = np.arctan2(-1*V,U)*180/np.pi          # Instantaneous relative direction of the wind speed vector (deg)
    wspd = np.sqrt(U**2+V**2+W**2) 

    us = U[0:nf]
    vs = V[0:nf]
    ws = W[0:nf]
    Ts = T[0:nf]
    qs = Q[0:nf]
    cs = C[0:nf]
    wdirs = wdirr[0:nf]
         
    # Means for FFT segment (Note that they differ from hourly means, see above):
    usm = us.mean()
    vsm = vs.mean()
    wsm = ws.mean()
    Tsm = Ts.mean()
    qsm = qs.mean()
    csm = cs.mean()
    wdirsm = wdirs.mean()

    # >>> Perform Fast Fourier Transform (FFT) to compute power spectra and cospectra:
    # In the following version linear detrend is used (no overlaping)

    F,su = signal.welch(us-usm,10,signal.windows.hamming(nf),detrend='linear') # (psd = Power Spectral Density)
    F,sv = signal.welch(vs-vsm,10,signal.windows.hamming(nf),detrend='linear')
    F,sw = signal.welch(ws-wsm,10,signal.windows.hamming(nf),detrend='linear')
    F,sT = signal.welch(Ts-Tsm,10,signal.windows.hamming(nf),detrend='linear')
    F,sq = signal.welch(qs-qsm,10,signal.windows.hamming(nf),detrend='linear')
    F,sc = signal.welch(cs-csm,10,signal.windows.hamming(nf),detrend='linear')
    F,swu = signal.csd(ws-wsm,us-usm,10,signal.windows.hamming(nf),detrend='linear')   # (csd = Cross Spectral Density)
    F,swv = signal.csd(ws-wsm,vs-vsm,10,signal.windows.hamming(nf),detrend='linear')
    F,swT = signal.csd(ws-wsm,Ts-Tsm,10,signal.windows.hamming(nf),detrend='linear')
    F,swq = signal.csd(ws-wsm,qs-qsm,10,signal.windows.hamming(nf),detrend='linear')
    F,swc = signal.csd(ws-wsm,cs-csm,10,signal.windows.hamming(nf),detrend='linear')
            
    # In addition to Chris' original code, computation of CuT, CvT, and Cuv are added by AG
    # Cospectra CuT & CvT are associated with the horizontal heat flux
    F,suT = signal.csd(us-usm,Ts-Tsm,10,signal.windows.hamming(nf),detrend='linear')
    F,svT = signal.csd(vs-vsm,Ts-Tsm,10,signal.windows.hamming(nf),detrend='linear')
    F,suv = signal.csd(us-usm,vs-vsm,10,signal.windows.hamming(nf),detrend='linear')
    F,suq = signal.csd(us-usm,qs-qsm,10,signal.windows.hamming(nf),detrend='linear')
    F,svq = signal.csd(vs-vsm,qs-qsm,10,signal.windows.hamming(nf),detrend='linear')
    F,suc = signal.csd(us-usm,cs-csm,10,signal.windows.hamming(nf),detrend='linear')
    F,svc = signal.csd(vs-vsm,cs-csm,10,signal.windows.hamming(nf),detrend='linear')
    

    # Also spectrum of wind speed direction is added (AG)
    F,swdir = signal.welch(wdirs-wdirsm,10,signal.windows.hamming(nf),detrend='linear')

    # Spectra smoothing
    nfd2 = nf/2
    c1   = 0.1
    jx   = 0
    dx   = 1
    ix   = 0
    inx  = 0
           
    sus    = np.zeros(int(nfd2), dtype=complex)
    svs    = np.zeros(int(nfd2), dtype=complex)
    sws    = np.zeros(int(nfd2), dtype=complex)
    sTs    = np.zeros(int(nfd2), dtype=complex)
    sqs    = np.zeros(int(nfd2), dtype=complex)
    scs    = np.zeros(int(nfd2), dtype=complex)
    cwus   = np.zeros(int(nfd2), dtype=complex)
    cwvs   = np.zeros(int(nfd2), dtype=complex)
    cwTs   = np.zeros(int(nfd2), dtype=complex)
    cuTs   = np.zeros(int(nfd2), dtype=complex)
    cvTs   = np.zeros(int(nfd2), dtype=complex)
    cwqs   = np.zeros(int(nfd2), dtype=complex)
    cuqs   = np.zeros(int(nfd2), dtype=complex)
    cvqs   = np.zeros(int(nfd2), dtype=complex)
    cwcs   = np.zeros(int(nfd2), dtype=complex)
    cucs   = np.zeros(int(nfd2), dtype=complex)
    cvcs   = np.zeros(int(nfd2), dtype=complex)
    cuvs   = np.zeros(int(nfd2), dtype=complex)
    swdirs = np.zeros(int(nfd2), dtype=complex)
    fs     = np.zeros(int(nfd2), dtype=complex)
    dfs    = np.zeros(int(nfd2), dtype=complex)

    while jx<nfd2:
        dx     = (dx*np.exp(c1))
        d1     = np.int32(np.floor(dx))
        acu    = 0
        acv    = 0
        acw    = 0
        acT    = 0
        acq    = 0
        acc    = 0
        acwu   = 0
        acwv   = 0
        acwT   = 0
        acuT   = 0
        acvT   = 0
        acwq   = 0
        acuq   = 0
        acvq   = 0
        acwc   = 0
        acuc   = 0
        acvc   = 0
        acuv   = 0
        acwdir = 0
        ac2    = 0
        k=0
        for jx in range(ix,ix+d1):
            if (jx==nfd2):
                break
            acu    = acu+su[jx]
            acv    = acv+sv[jx]
            acw    = acw+sw[jx]
            acT    = acT+sT[jx]
            acq    = acq+sq[jx]
            acc    = acc+sc[jx]
            acwu   = acwu+swu[jx]
            acwv   = acwv+swv[jx]
            acwT   = acwT+swT[jx]
            acuT   = acuT+suT[jx]
            acvT   = acvT+svT[jx]
            acwq   = acwq+swq[jx]
            acuq   = acuq+suq[jx]
            acvq   = acvq+svq[jx]
            acwc   = acwc+swc[jx]
            acuc   = acuc+suc[jx]
            acvc   = acvc+svc[jx]
            acuv   = acuv+suv[jx]
            acwdir = acwdir+swdir[jx]
            ac2    = ac2+F[jx]
            k=k+1    

        sus[inx] = acu/k   
        svs[inx] = acv/k   
        sws[inx] = acw/k   
        sTs[inx] = acT/k   
        sqs[inx] = acq/k 
        scs[inx] = acc/k 
        cwus[inx] = acwu/k 
        cwvs[inx] = acwv/k
        cwTs[inx] = acwT/k
        cuTs[inx] = acuT/k
        cvTs[inx] = acvT/k
        cwqs[inx] = acwq/k
        cuqs[inx] = acuq/k
        cvqs[inx] = acvq/k
        cwcs[inx] = acwc/k
        cucs[inx] = acuc/k
        cvcs[inx] = acvc/k
        cuvs[inx] = acuv/k
        swdirs[inx]=acwdir/k
        fs[inx]=ac2/k
        dfs[inx]=F[jx]-F[ix]+F[1]
        ix=jx+1
        inx=inx+1
    
    sus = sus[0:inx]
    svs = svs[0:inx]
    sws = sws[0:inx]
    sTs = sTs[0:inx]
    sqs = sqs[0:inx]
    scs = scs[0:inx]
    cwus = cwus[0:inx]
    cwvs = cwvs[0:inx]
    cwTs = cwTs[0:inx]
    cuTs = cuTs[0:inx]
    cvTs = cvTs[0:inx]
    cwqs = cwqs[0:inx]
    cuqs = cuqs[0:inx]
    cvqs = cvqs[0:inx]
    cwcs = cwcs[0:inx]
    cucs = cucs[0:inx]
    cvcs = cvcs[0:inx]
    cuvs = cuvs[0:inx]
    swdirs = swdirs[0:inx]
    fs = fs[0:inx]
    dfs = dfs[0:inx]
    
    # take the real part
    sus    = np.real(sus)
    svs    = np.real(svs)
    sws    = np.real(sws)
    sTs    = np.real(sTs)
    sqs    = np.real(sqs)
    scs    = np.real(scs)
    cwus   = np.real(cwus)
    cwvs   = np.real(cwvs)
    cwTs   = np.real(cwTs)
    cuTs   = np.real(cuTs)
    cvTs   = np.real(cvTs)
    cwqs   = np.real(cwqs)
    cuqs   = np.real(cuqs)
    cvqs   = np.real(cvqs)
    cwcs   = np.real(cwcs)
    cucs   = np.real(cucs)
    cvcs   = np.real(cvcs)
    cuvs   = np.real(cuvs)
    swdirs = np.real(swdirs)
    fs     = np.real(fs)
    dfs    = np.real(dfs)

    #+++++++++++++++++++++++++++++ Wind speed and direction ++++++++++++++++++++++++++++++++++++
    wsp = (um**2 + vm**2 + wm**2)**0.5       # hour averaged wind speed (m/s)
    wdir = np.arctan2(-vm,um)*180/np.pi      # averaged relative wind direction (deg)

    # Convert wind direction range to [0 360] segment:
    if (wdir < 0):
        wdir = wdir + 360

    #++++++++++++++++++++++++++++ Fluxes Based on the Cospectra +++++++++++++++++++++++++++++++
    # >>> Compute covariances ("fluxes") based on the appropriate cospectra integration (the
    # total flux is the sum of cospectra frequency components) The turbulent fluxes and
    # variances are derived through frequency integration of the appropriate cospectra and spectra
    cwux = np.sum(cwus*dfs)
    cwvx = np.sum(cwvs*dfs)
    cuvx = np.sum(cuvs*dfs)
    cwTx = np.sum(cwTs*dfs)
    cuTx = np.sum(cuTs*dfs)
    cvTx = np.sum(cvTs*dfs)
    cwqx = np.sum(cwqs*dfs)
    cuqx = np.sum(cuqs*dfs)
    cvqx = np.sum(cvqs*dfs)
    cwcx = np.sum(cwcs*dfs)
    cucx = np.sum(cucs*dfs)
    cvcx = np.sum(cvcs*dfs)
    # >>> Compute standard deviations based on the appropriate spectra integration:
    sigu_spc = (np.sum(sus*dfs))**0.5       # based on the Su-spectra integration
    sigv_spc = (np.sum(svs*dfs))**0.5       # based on the Sv-spectra integration
    sigw_spc = (np.sum(sws*dfs))**0.5       # based on the Sw-spectra integration
    sigT_spc = (np.sum(sTs*dfs))**0.5       # based on the St-spectra integration
    sigq_spc = (np.sum(sqs*dfs))**0.5       # based on the St-spectra integration
    sigc_spc = (np.sum(scs*dfs))**0.5       # based on the St-spectra integration
    sigwdir_spc = (np.sum(swdirs*dfs))**0.5 # based on the Sdir-spectra integration
    ssm = (usm*usm+vsm*vsm)**0.5

    # !! Note that sigwdir_spc may be used for quality control (or QC for short) of the data
    # For example sigwdir_spc > 15 deg may be considered as "bad" data (non-stationary)
    # However this QC is not applicable for free convection limit (light winds)
    
    # >>> Wind stress components and the friction velocity:
    wu_csp = cwux   # wu-covariance based on the wu-cospectra integration
    wv_csp = cwvx   # wv-covariance based on the wv-cospectra integration
    uv_csp = cuvx   # uv-covariance based on the uv-cospectra integration
    #ustar = (np.abs(wu_csp))**0.5 # the friction velocity based only on the downstream, uw,  stress components (m/s)
    #ustar = -np.sign(wu_csp)*((abs(wu_csp))**2 + (abs(wv_csp))**2)**(1/4)  # ustar is based on both stress components (m/s)
    ustar = (wu_csp**2+wv_csp**2)**(1/4)

    # >>> Fluxes:
    # Note that a sonic anemometer/thermometer measures the so-called 'sonic' temperature which is close to the virtual temperature
    # Thus <w't'> here is a buoyancy flux
    # Moisture correction is necessary to obtain the correct sensible heat flux (e.g., Kaimal and Finnigan, 1994)
    # However, in Arctic and Antarctic this value very close to the temperature flux (i.e. the sensible heat flux)
    wT_csp = cwTx   # wt-covariance, vertical flux of the sonic temperature  [deg m/s]
    uT_csp = cuTx   # ut-covariance, horizontal flux of the sonic temperature (along-wind component) [deg m/s]
    vT_csp = cvTx   # vt-covariance, horizontal flux of the sonic temperature (cross-wind component) [deg m/s]
    wq_csp = cwqx
    uq_csp = cuqx
    vq_csp = cvqx
    wc_csp = cwcx
    uc_csp = cucx
    vc_csp = cvcx


    # -- calculate sensible heat flux --
    #
    # Nominal equation (e.g., MOSAiC): Hs = wT_csp*rho*cp  
    #
    # Metek output is sonic temperature, which includes contributions from variance in temperature and also density variance caused by moisture fluxes (letent heat).
    # Thus, Hs = wT_csp*rho*cp yields a buoyancy flux, not a heat flux. Generally, the moisture component should be removed when the Bowen ratio (B) is < 1 
    # (see https://www.eol.ucar.edu/content/corrections-sensible-and-latent-heat-flux-measurements) (hense, no correction for MOSAiC). For SPLASH, because we do not have continuous or 
    # continuously-reliable Licor data, we cannot subtract the moisture density term off using wq_csp for most instances. At SPLASH, B is < 1 ~36% of the time. 
    # If you can estimate B, then you can correct Hs using only the sonic following Schotanus et al. (1983, https://doi.org/10.1007/BF00164332). 
    # We do so following equation 15 in De Bruin & Holstag (1982, https://doi.org/10.1007/BF00164332) where equations for s and lamba are found here:
    # https://www.fao.org/3/x0490e/x0490e0j.htm#annex%202.%20meteorological%20tables.   
    c2            = 403.*(temp+tdk)*(1+0.51*(mr/(1+mr)))                                      # speed of sound
    gamma         = ((cp/1e6)*pr/10)/((Le/1e6)*0.622)                                         # psychrometric constant [kPa/C]
    Dslope        = 4098*(0.6108*np.exp((17.27*temp)/(temp+237.3))/(temp+237.3)**2)           # slope of the saturation vapor pressure temperature curve
    Bowen         = gamma/Dslope                                                              # Bowen ratio                                           
    term1         = 1+((0.51*(temp+tdk)*cp)/(Le*Bowen))                                       # Eq. (10) Schotanus       
    term2         = 2*(((temp+tdk)*wspd.mean())/c2)*wu_csp                                    # Eq. (10) Schotanus    
    Hs            = ((wT_csp+term2)/term1)*cp*rho                                             # Eq. (10) Schotanus with conversion to Wm**2
    Tstar         = -wT_csp/np.abs(ustar)                                                     # the temperature scale

    # -- calculate sensible heat flux --    
    Hl_raw        = wq_csp*Le*rho                                                             # latent heat flux (W/m2)
    Hl            = Hl_raw+(Le*qsm*(1.61*wq_csp/rho_d+(1+1.61*sigma)*wT_csp/(temp+tdk)))      # Webb (1980, eq. 24) corrected
    
    # -- calculate co2 flux --      
    CO2_flux_raw  = wc_csp                                                                    # co2 mass flux (mg m^-2 s^-1)
    CO2_flux      = CO2_flux_raw + (csm*(1.61*wq_csp/rho_d+(1+1.61*sigma)*wT_csp/(temp+tdk))) # Webb (1980, eq. 24) corrected                                             
    CO2_flux_Wm2  = -0.479*CO2_flux/1000*0.022722366761722*1e6                                # photsynthetic flux Grachev et al. (2020) https://www.sciencedirect.com/science/article/pii/S0168192319304393
    
    # The loop below is added for cases when number of points < 32768 (54.61333 min) or < 16384 (27.30667 min)
    # It defines frequencies for the inertial subrange
    # !! -1 to get to 0 based (python) but preserving the original values for reference.
    if npt < 8192:  # 2^12 = 4096 points  (or = 4096/10 sec  =  6.82667 min for samp_freq = 10 Hz)
       fsl = 4-1
       fsn = 53-1
       fsi01 = 34-1
       fsi02 = 35-1
       fsi03 = 36-1
       fsi04 = 37-1
       fsi05 = 38-1
       fsi06 = 39-1
       fsi07 = 40-1
       fsi08 = 41-1
       fsi09 = 42-1
       fsi10 = 43-1
       fsi11 = 44-1
       fsi12 = 45-1

    elif npt < 16384: # 2^13 = 8192 points  (or =  8192/10 sec = 13.65333 min for samp_freq = 10 Hz)
       fsl = 6-1
       fsn = 60-1
       fsi01 = 40-1
       fsi02 = 41-1
       fsi03 = 42-1
       fsi04 = 43-1
       fsi05 = 44-1
       fsi06 = 45-1
       fsi07 = 46-1
       fsi08 = 47-1
       fsi09 = 48-1
       fsi10 = 49-1
       fsi11 = 50-1
       fsi12 = 51-1

    elif npt < 32768: # 2^14 = 16384 points (or = 16384/10 sec = 27.30667 min for samp_freq = 10 Hz)
       fsl = 9-1
       fsn = 67-1
       fsi01 = 47-1
       fsi02 = 48-1
       fsi03 = 49-1
       fsi04 = 50-1
       fsi05 = 51-1
       fsi06 = 52-1
       fsi07 = 53-1
       fsi08 = 54-1
       fsi09 = 55-1
       fsi10 = 56-1
       fsi11 = 57-1
       fsi12 = 58-1

    else: # 2^15 = 32768 points (or = 32768/10 sec = 54.61333 min for samp_freq = 10 Hz)
       fsl = 13-1
       fsn = 74-1
       fsi01 = 54-1
       fsi02 = 55-1
       fsi03 = 56-1
       fsi04 = 57-1
       fsi05 = 58-1
       fsi06 = 59-1
       fsi07 = 60-1
       fsi08 = 61-1
       fsi09 = 62-1
       fsi10 = 63-1
       fsi11 = 64-1
       fsi12 = 65-1
  
    # >>> Some Monin-Obukhov (MO) parameters based on the local turbulent measurements:
    # Monin-Obukhov stability parameter, z/L:
    zeta_level_n = - ((0.4*9.81)/(Tsm+tdk))*(z_level_n*wT_csp/(ustar**3))
    # Drag coefficient, Cd:
    Cd = ustar**2/wsp**2 #- wu_csp/(wsp**2)
    # MO universal functions for the standard deviations:
    phi_u = sigu_spc/ustar
    phi_v = sigv_spc/ustar
    phi_w = sigw_spc/ustar
    phi_T = sigT_spc/np.abs(Tstar)
    # MO universal function for the horizontal heat flux:
    phi_uT = uT_csp/wT_csp
    
    #+++++++++++++++++++ Compute structure parameters in the inertial subrange +++++++++++++++++
    #+++++++++++++++++++++++++++++ 5/3 Kolmogorov power law +++++++++++++++++++++++++++++++++++

    # Structure parameters are used to derive dissipation rate of the turbulent kinetic energy ('epsilon') and half the temperature variance (Nt)
    # The inertial subrange is considered in the frequency subrange ~ 0.6-2 Hz
    # Generally this subrange depends on a height of measurements and may be different for w-component
    # For 2^15 = 32768 data points = 54.61333 min, the structure parameters are computed in the frequency domain between the 54th and 65th spectral values:
    #    fs(fsi01:fsi12) = fs(54:65) = [0.6662 0.7372 0.8156 0.9023 0.9981 1.1041 1.2213 1.3507 1.4937 1.6518 1.8265 2.0197]
    # For 2^14 = 16384 data points = 27.30667 min, the structure parameters are computed in the frequency domain between the 47th and 58th spectral values:
    #    fs(fsi01:fsi12) = fs(47:58) = [0.6531 0.7233 0.8011 0.8871 0.9824 1.0876 1.2039 1.3324 1.4743 1.6312 1.8045 1.9962]
    # For 2^13 =  8192 data points = 13.65333 min, the structure parameters are computed in the frequency domain between the 40th and 51st spectral values:
    #    fs(fsi01:fsi12) = fs(40:51) = [0.6342 0.7037 0.7806 0.8655 0.9595 1.0638 1.1792 1.3062 1.4465 1.6022 1.7743 1.9647]
    # For 2^12 =  4096 data points =  6.82667 min, the structure parameters are computed in the frequency domain between the 34th and 45th spectral values:
    #    fs(fsi01:fsi12) = fs(34:45) = [0.6738 0.7495 0.8337 0.9265 1.0291 1.1426 1.2683 1.4075 1.5613 1.7310 1.9189 2.1277]

    # Structure parameters definition:
    # Cu^2 = 4*alpha*epsilon^{2/3} (alpha=0.55); Ct^2 = 4*beta*Nt*epsilon^{-1/3} (beta=0.8)
    # Relationships between the structure parameters and the frequency spectra:
    # Cu^2 = 4*(2pi/U)^{2/3}*Su(f)*f^{-5/3}; Ct^2 = 4*(2pi/U)^{2/3}*St(f)*f^{-5/3};
    # Dimensions: [Cu^2] = [Ustar^2/z^{2/3}]; [Ct^2] = [Tstar^2/z^{2/3}];
    # Local isotropy in the inertial subrange assumes:
    # Fv(k)=Fw(k)=(4/3)Fu(k), or Sv(f)=Sw(f)=(4/3)Su(f), or Cv^2=Cw^2=(4/3)Cu^2.
    # Conversion from wavenumber to frequency scales: k*Fu(k)=f*Su(f).
    # See details in the textbook by J.C. Kaimal & J.J. Finnigan "Atmospheric Boundary Layer Flows" (1994)

    gfac = 4*(2*np.pi/wsp)**0.667
    cu2 = gfac*np.median(sus[fsi01:fsi12]*fs[fsi01:fsi12]**1.667)               # Cu^2, U structure parameter [x^2/m^2/3 = m^1.33/s^2]
    cv2 = gfac*np.median(svs[fsi01:fsi12]*fs[fsi01:fsi12]**1.667)               # Cv^2, V structure parameter [x^2/m^2/3 = m^1.33/s^2]
    cw2 = gfac*np.median(sws[fsi01:fsi12]*fs[fsi01:fsi12]**1.667)               # Cw^2, W structure parameter [x^2/m^2/3 = m^1.33/s^2]
    cT2 = gfac*np.median((sTs[fsi01:fsi12]-np.min(sTs))*fs[fsi01:fsi12]**1.667) # Ct^2, T structure parameter [x^2/m^2/3 = deg^2/m^2/3]; (term -min(sTs) reduces some noise - proposed by Chris F.)
    # cT2 = gfac*numpy.median((sTs[fsi01:fsi12])*fs[fsi01:fsi12]**1.667)          # Ct^2, T structure parameter [x^2/m^2/3 = deg^2/m^2/3]


    #+++++++++++++++++ Compute dissipation rate of the turbulent kinetic energy ++++++++++++++++

    # Non-dimensional dissipation rate of the turbulent kinetic energy (TKE):
    # Phi_epsilon = kappa*z*epsilon/Ustar^3 - see Eq. (1.30) in Kaimal & Finnigan (1994)
    # Substitution of epsilon = (Cu^2/4*alpha)^(3/2) leads to
    # Phi_epsilon = kappa*[(z^2/3)*Cu^2/4*alpha*Ustar^2] = kappa*(Cu^2n/4*alpha)^3/2, where

    alphaK = 0.55 # - the Kolmogorov constant (alpha=0.55 - see above)
    # Note that Cu^2=(3/4)Cv^2=(3/4)Cw^2

    # >>> For epsilon_u (Based on Su) ++++++++++++++++++++++++++++++++++++++
    # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the longitudinal velocity component in the inertial subrange [m^2/s^3]:
    epsilon_u = (cu2/(4*alphaK))**(3/2)
    # >>> For epsilon_v (Based on Sv) ++++++++++++++++++++++++++++++++++++++
    # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the lateral velocity component in the inertial subrange [m^2/s^3]:
    epsilon_v = (3/4)*(cv2/(4*alphaK))**(3/2)
    # >>> For epsilon_w (Based on Sw) ++++++++++++++++++++++++++++++++++++++
    # Dissipation rate of the turbulent kinetic energy based on the energy spectra of the vertical velocity component in the inertial subrange [m^2/s^3]:
    epsilon_w = (3/4)*(cw2/(4*alphaK))**(3/2)
    # >>> Median epsilon (median of epsilon_u, epsilon_v, & epsilon_w) ++++++++++++++++++++++++++++++++++++++
    # Dissipation rate of the turbulent kinetic energy = median of the values derived from Su, Sv, & Sw [m^2/s^3]:
    epsilon = np.median([epsilon_u,epsilon_v,epsilon_w])
    # Monin-Obukhov universal function Phi_epsilon based on the median epsilon:
    Phi_epsilon = (0.4*z_level_n*epsilon)/(ustar**3)
    
    #+++++ Compute the dissipation (destruction) rate for half the temperature variance +++++++
    #++++++++++++++++++ 5/3 Obukhov-Corrsin power law for the passive scalar ++++++++++++++++++

    #  Temperature Structure Parameter: Ct^2 = 4*beta*Nt*epsilon^{-1/3} (beta=0.8)
    betaK = 0.8 # - the Kolmogorov (Obukhov-Corrsin) constant for the passive scalar (beta=0.8 - see above)

    # >>> Nt (Based on Ct^2 derived from St) ++++++++++++++++++++++++++++++++++++++
    # The dissipation (destruction) rate for half the temperature variance [deg^2/s]:
    Nt = (cT2*(epsilon**(1/3)))/(4*betaK)
    # Monin-Obukhov universal function Phi_Nt:
    Phi_Nt = (0.4*z_level_n*Nt)/(ustar*Tstar**2)

    #++++++++++++++++++++++ Compute spectral slopes in the inertial subrange +++++++++++++++++++

    # >>> Spectral slopes are computed in the frequency domain defined above
    # Compute a spectral slope in the inertial subrange for Su. Individual slopes:
    nSu_1 = np.log(sus[fsi07]/sus[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSu_2 = np.log(sus[fsi08]/sus[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSu_3 = np.log(sus[fsi09]/sus[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSu_4 = np.log(sus[fsi10]/sus[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSu_5 = np.log(sus[fsi11]/sus[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSu_6 = np.log(sus[fsi12]/sus[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSu = np.median([nSu_1,nSu_2,nSu_3,nSu_4,nSu_5,nSu_6])

    # Compute a spectral slope in the inertial subrange for Sv.
    # Individual slopes:
    nSv_1 = np.log(svs[fsi07]/svs[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSv_2 = np.log(svs[fsi08]/svs[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSv_3 = np.log(svs[fsi09]/svs[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSv_4 = np.log(svs[fsi10]/svs[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSv_5 = np.log(svs[fsi11]/svs[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSv_6 = np.log(svs[fsi12]/svs[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSv = np.median([nSv_1,nSv_2,nSv_3,nSv_4,nSv_5,nSv_6])

    # Compute a spectral slope in the inertial subrange for Sw. Individual slopes:
    nSw_1 = np.log(sws[fsi07]/sws[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSw_2 = np.log(sws[fsi08]/sws[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSw_3 = np.log(sws[fsi09]/sws[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSw_4 = np.log(sws[fsi10]/sws[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSw_5 = np.log(sws[fsi11]/sws[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSw_6 = np.log(sws[fsi12]/sws[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSw = np.median([nSw_1,nSw_2,nSw_3,nSw_4,nSw_5,nSw_6])

    # Compute a spectral slope in the inertial subrange for St.
    # Individual slopes:
    nSt_1 = np.log(sTs[fsi07]/sTs[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSt_2 = np.log(sTs[fsi08]/sTs[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSt_3 = np.log(sTs[fsi09]/sTs[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSt_4 = np.log(sTs[fsi10]/sTs[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSt_5 = np.log(sTs[fsi11]/sTs[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSt_6 = np.log(sTs[fsi12]/sTs[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSt = np.median([nSt_1,nSt_2,nSt_3,nSt_4,nSt_5,nSt_6])
    
    # Compute a spectral slope in the inertial subrange for St.
    # Individual slopes:
    nSq_1 = np.log(sqs[fsi07]/sqs[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSq_2 = np.log(sqs[fsi08]/sqs[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSq_3 = np.log(sqs[fsi09]/sqs[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSq_4 = np.log(sqs[fsi10]/sqs[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSq_5 = np.log(sqs[fsi11]/sqs[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSq_6 = np.log(sqs[fsi12]/sqs[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSq = np.median([nSq_1,nSq_2,nSq_3,nSq_4,nSq_5,nSq_6])
    
    # Compute a spectral slope in the inertial subrange for St.
    # Individual slopes:
    nSc_1 = np.log(scs[fsi07]/sqs[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSc_2 = np.log(scs[fsi08]/sqs[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSc_3 = np.log(scs[fsi09]/sqs[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSc_4 = np.log(scs[fsi10]/sqs[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSc_5 = np.log(scs[fsi11]/sqs[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSc_6 = np.log(scs[fsi12]/sqs[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSc = np.median([nSc_1,nSc_2,nSc_3,nSc_4,nSc_5,nSc_6])
    
    # Compute a spectral slope in the inertial subrange for Swdirs (wind direction spectrum).
    # Individual slopes:
    nSwdir_1 = np.log(swdirs[fsi07]/swdirs[fsi01])/np.log(fs[fsi07]/fs[fsi01])
    nSwdir_2 = np.log(swdirs[fsi08]/swdirs[fsi02])/np.log(fs[fsi08]/fs[fsi02])
    nSwdir_3 = np.log(swdirs[fsi09]/swdirs[fsi03])/np.log(fs[fsi09]/fs[fsi03])
    nSwdir_4 = np.log(swdirs[fsi10]/swdirs[fsi04])/np.log(fs[fsi10]/fs[fsi04])
    nSwdir_5 = np.log(swdirs[fsi11]/swdirs[fsi05])/np.log(fs[fsi11]/fs[fsi05])
    nSwdir_6 = np.log(swdirs[fsi12]/swdirs[fsi06])/np.log(fs[fsi12]/fs[fsi06])
    # Median spectral slope in the inertial subrange:
    nSwdir = np.median([nSwdir_1,nSwdir_2,nSwdir_3,nSwdir_4,nSwdir_5,nSwdir_6])


    # Note that the spectral slopes can be used as QC parameters, e.g. as indicator of resolution limit of a sonic anemometer
    # This resolution leads to a step ladder appearance in the data time series, and the turbulent fluxes cannot be reliably calculated
    # See Fig. 1b in Vickers & Mahrt (1997)J. Atmos. Oc. Tech. 14(3): 512�526
    # Eventually, in the very stable regime, the spectral slope levels off asymptotically at zero which corresponds to 'white noise' in the sensor
    # See spectra at level 5 in Fig. 3 in Grachev et al. (2008) Acta Geophysica. 56(1): 142�166.
    # For example, conditions nSu, nSv, nSu, and nSt > - 0.5 (or even > - 1) can be used as QC thresholds

    #+++++++++++++++Second-order moments of atmospheric turbulence (including fluxes)+++++++++++
    #+++++++++++++++ (Direct variances and covariances based on Reynolds averaging) ++++++++++++

    # Removing linear trends from the data (detrending the data):
    us_dtr = signal.detrend(us)
    vs_dtr = signal.detrend(vs)
    ws_dtr = signal.detrend(ws)
    Ts_dtr = signal.detrend(Ts)
    qs_dtr = signal.detrend(qs)
    cs_dtr = signal.detrend(cs)

    # Trend lines:
    us_trend = us - us_dtr
    vs_trend = vs - vs_dtr
    ws_trend = ws - ws_dtr
    Ts_trend = Ts - Ts_dtr
    qs_trend = qs - qs_dtr
    cs_trend = cs - cs_dtr

    # Mean of the detrended data (should be very close to zero)
    us_dtr_m = us_dtr.mean()
    vs_dtr_m = vs_dtr.mean()
    ws_dtr_m = ws_dtr.mean()
    Ts_dtr_m = Ts_dtr.mean()
    qs_dtr_m = qs_dtr.mean()
    cs_dtr_m = cs_dtr.mean()

    # >>> Covariances and Variances of the detrending data:
    cov_wu   = np.cov(ws_dtr,us_dtr)
    cov_wv   = np.cov(ws_dtr,vs_dtr)
    cov_uv   = np.cov(us_dtr,vs_dtr)
    cov_wT   = np.cov(ws_dtr,Ts_dtr)
    cov_uT   = np.cov(us_dtr,Ts_dtr)
    cov_vT   = np.cov(vs_dtr,Ts_dtr)
    cov_wq   = np.cov(ws_dtr,qs_dtr)
    cov_uq   = np.cov(us_dtr,qs_dtr)
    cov_vq   = np.cov(vs_dtr,qs_dtr)
    cov_wc   = np.cov(ws_dtr,cs_dtr)
    cov_uc   = np.cov(us_dtr,cs_dtr)
    cov_vc   = np.cov(vs_dtr,cs_dtr)
    wu_cov   = cov_wu[0,1]
    wv_cov   = cov_wv[0,1]
    uv_cov   = cov_uv[0,1]
    wT_cov   = cov_wT[0,1]
    uT_cov   = cov_uT[0,1]
    vT_cov   = cov_vT[0,1]
    wq_cov   = cov_wq[0,1]
    uq_cov   = cov_uq[0,1]
    vq_cov   = cov_vq[0,1]
    wc_cov   = cov_wc[0,1]
    uc_cov   = cov_uc[0,1]
    vc_cov   = cov_vc[0,1]
    sigu_cov = cov_wu[1,1]    # Standard deviation of the u-wind component
    sigv_cov = cov_wv[1,1]    # Standard deviation of the v-wind component
    sigw_cov = cov_wu[0,0]    # Standard deviation of the w-wind component
    sigT_cov = cov_wT[1,1]    # Standard deviation of the the air temperature (sonic temperature)
    sigq_cov = cov_wq[1,1]    # 
    sigc_cov = cov_wc[1,1]    # 
    sigu_std = np.std(us_dtr) # Standard deviation of the u-wind component
    sigv_std = np.std(vs_dtr) # Standard deviation of the v-wind component
    sigw_std = np.std(ws_dtr) # Standard deviation of the w-wind component
    sigT_std = np.std(Ts_dtr) # Standard deviation of the the air temperature (sonic temperature)
    sigq_std = np.std(qs_dtr) # 
    sigc_std = np.std(cs_dtr) # 

    # >>> Second-order moments (directly or by definition)
    # Linear detrending but no block-averaging of the data!
    # >>> Note that this is the same as 'cov' MatLab command above
    # nf = 32768 (54.6133 min averaged data)
    uu_dtr   = (np.sum(us_dtr*us_dtr))/nf
    uv_dtr   = (np.sum(us_dtr*vs_dtr))/nf
    uw_dtr   = (np.sum(us_dtr*ws_dtr))/nf
    vv_dtr   = (np.sum(vs_dtr*vs_dtr))/nf
    vw_dtr   = (np.sum(vs_dtr*ws_dtr))/nf
    ww_dtr   = (np.sum(ws_dtr*ws_dtr))/nf
    uT_dtr   = (np.sum(us_dtr*Ts_dtr))/nf
    vT_dtr   = (np.sum(vs_dtr*Ts_dtr))/nf
    wT_dtr   = (np.sum(ws_dtr*Ts_dtr))/nf
    TT_dtr   = (np.sum(Ts_dtr*Ts_dtr))/nf
    uq_dtr   = (np.sum(us_dtr*qs_dtr))/nf
    vq_dtr   = (np.sum(vs_dtr*qs_dtr))/nf
    wq_dtr   = (np.sum(ws_dtr*qs_dtr))/nf
    qq_dtr   = (np.sum(qs_dtr*qs_dtr))/nf
    uc_dtr   = (np.sum(us_dtr*cs_dtr))/nf
    vc_dtr   = (np.sum(vs_dtr*cs_dtr))/nf
    wc_dtr   = (np.sum(ws_dtr*cs_dtr))/nf
    cc_dtr   = (np.sum(cs_dtr*cs_dtr))/nf
    sigu_dtr = (uu_dtr)**0.5 # Standard deviation of the u-wind component
    sigv_dtr = (vv_dtr)**0.5 # Standard deviation of the v-wind component
    sigw_dtr = (ww_dtr)**0.5 # Standard deviation of the w-wind component
    sigT_dtr = (TT_dtr)**0.5 # Standard deviation of the the air temperature (sonic temperature)
    sigq_dtr = (qq_dtr)**0.5 # Standard deviation of the the air temperature (sonic temperature)
    sigc_dtr = (cc_dtr)**0.5 # Standard deviation of the the air temperature (sonic temperature)

    # Note that sigu_cov = sigu_std = sigu_dtr (the same for 'v', 'w', and 'T') - different methods of computation

    # >>> Standard deviations of the detrending second-order moments (fluxes and variances):
    sig_uv_dtr = np.std(us_dtr*vs_dtr)
    sig_uw_dtr = np.std(us_dtr*ws_dtr)
    sig_vw_dtr = np.std(vs_dtr*ws_dtr)
    sig_uT_dtr = np.std(us_dtr*Ts_dtr)
    sig_vT_dtr = np.std(vs_dtr*Ts_dtr)
    sig_wT_dtr = np.std(ws_dtr*Ts_dtr)
    sig_uq_dtr = np.std(us_dtr*qs_dtr)
    sig_vq_dtr = np.std(vs_dtr*qs_dtr)
    sig_wq_dtr = np.std(ws_dtr*qs_dtr)
    sig_uc_dtr = np.std(us_dtr*cs_dtr)
    sig_vc_dtr = np.std(vs_dtr*cs_dtr)
    sig_wc_dtr = np.std(ws_dtr*cs_dtr)
    sig_uu_dtr = np.std(us_dtr*us_dtr)
    sig_vv_dtr = np.std(vs_dtr*vs_dtr)
    sig_ww_dtr = np.std(ws_dtr*ws_dtr)
    sig_TT_dtr = np.std(Ts_dtr*Ts_dtr)
    sig_qq_dtr = np.std(qs_dtr*qs_dtr)
    sig_cc_dtr = np.std(cs_dtr*cs_dtr)

    # >>> Second-order moments (directly or by definition)
    # Same as above, but no detrending and no block-averaging of the data!
    # nf = 32768 (54.6133 min averaged data)
    uu_direct   = (np.sum((us-usm)*(us-usm)))/nf
    uv_direct   = (np.sum((us-usm)*(vs-vsm)))/nf
    uw_direct   = (np.sum((us-usm)*(ws-wsm)))/nf
    vv_direct   = (np.sum((vs-vsm)*(vs-vsm)))/nf
    vw_direct   = (np.sum((vs-vsm)*(ws-wsm)))/nf
    ww_direct   = (np.sum((ws-wsm)*(ws-wsm)))/nf
    uT_direct   = (np.sum((us-usm)*(Ts-Tsm)))/nf
    vT_direct   = (np.sum((vs-vsm)*(Ts-Tsm)))/nf
    wT_direct   = (np.sum((ws-wsm)*(Ts-Tsm)))/nf
    TT_direct   = (np.sum((Ts-Tsm)*(Ts-Tsm)))/nf
    uq_direct   = (np.sum((us-usm)*(qs-qsm)))/nf
    vq_direct   = (np.sum((vs-vsm)*(qs-qsm)))/nf
    wq_direct   = (np.sum((ws-wsm)*(qs-qsm)))/nf
    qq_direct   = (np.sum((qs-qsm)*(qs-qsm)))/nf
    uc_direct   = (np.sum((us-usm)*(qs-qsm)))/nf
    vc_direct   = (np.sum((vs-vsm)*(cs-csm)))/nf
    wc_direct   = (np.sum((ws-wsm)*(cs-csm)))/nf
    cc_direct   = (np.sum((cs-csm)*(cs-csm)))/nf
    sigu_direct = (uu_direct)**0.5
    sigv_direct = (vv_direct)**0.5
    sigw_direct = (ww_direct)**0.5
    sigT_direct = (TT_direct)**0.5
    sigq_direct = (qq_direct)**0.5
    sigc_direct = (cc_direct)**0.5

    # Ratio of the fluxes (standard deviations) derived from covariances to the appropriate values derived from cospectra (spectra):
    wu_ratio   = wu_cov/wu_csp
    wv_ratio   = wv_cov/wv_csp
    wT_ratio   = wT_cov/wT_csp
    wq_ratio   = wq_cov/wq_csp
    wc_ratio   = wc_cov/wc_csp
    sigu_ratio = sigu_cov/sigu_spc
    sigv_ratio = sigv_cov/sigv_spc
    sigw_ratio = sigw_cov/sigw_spc
    sigT_ratio = sigT_cov/sigT_spc
    sigq_ratio = sigq_cov/sigq_spc
    sigc_ratio = sigc_cov/sigc_spc

    # See details of the linear detrending in:
    # Gash, J. H. C. and A. D. Culf. 1996. Applying linear de-trend to eddy correlation data in real time. Boundary-Layer Meteorology, 79: 301-306.

    #++++++++++++++++++++ Third-order moments of atmospheric turbulence ++++++++++++++++++++++++
    # Linear detrending but no block-averaging of the data!
    # nf = 32768 (54.6133 min averaged data)

    uuu_dtr = (np.sum(us_dtr*us_dtr*us_dtr))/nf
    uuv_dtr = (np.sum(us_dtr*us_dtr*vs_dtr))/nf
    uuw_dtr = (np.sum(us_dtr*us_dtr*ws_dtr))/nf
    uvv_dtr = (np.sum(us_dtr*vs_dtr*vs_dtr))/nf
    uvw_dtr = (np.sum(us_dtr*vs_dtr*ws_dtr))/nf
    uww_dtr = (np.sum(us_dtr*ws_dtr*ws_dtr))/nf
    vvv_dtr = (np.sum(vs_dtr*vs_dtr*vs_dtr))/nf
    vvw_dtr = (np.sum(vs_dtr*vs_dtr*ws_dtr))/nf
    vww_dtr = (np.sum(vs_dtr*ws_dtr*ws_dtr))/nf
    www_dtr = (np.sum(ws_dtr*ws_dtr*ws_dtr))/nf

    uuT_dtr = (np.sum(us_dtr*us_dtr*Ts_dtr))/nf
    uvT_dtr = (np.sum(us_dtr*vs_dtr*Ts_dtr))/nf
    uwT_dtr = (np.sum(us_dtr*ws_dtr*Ts_dtr))/nf
    vvT_dtr = (np.sum(vs_dtr*vs_dtr*Ts_dtr))/nf
    vwT_dtr = (np.sum(vs_dtr*ws_dtr*Ts_dtr))/nf
    wwT_dtr = (np.sum(ws_dtr*ws_dtr*Ts_dtr))/nf
    uTT_dtr = (np.sum(us_dtr*Ts_dtr*Ts_dtr))/nf
    vTT_dtr = (np.sum(vs_dtr*Ts_dtr*Ts_dtr))/nf
    wTT_dtr = (np.sum(ws_dtr*Ts_dtr*Ts_dtr))/nf
    TTT_dtr = (np.sum(Ts_dtr*Ts_dtr*Ts_dtr))/nf
    
    uuq_dtr = (np.sum(us_dtr*us_dtr*qs_dtr))/nf
    uvq_dtr = (np.sum(us_dtr*vs_dtr*qs_dtr))/nf
    uwq_dtr = (np.sum(us_dtr*ws_dtr*qs_dtr))/nf
    vvq_dtr = (np.sum(vs_dtr*vs_dtr*qs_dtr))/nf
    vwq_dtr = (np.sum(vs_dtr*ws_dtr*qs_dtr))/nf
    wwq_dtr = (np.sum(ws_dtr*ws_dtr*qs_dtr))/nf
    uqq_dtr = (np.sum(us_dtr*Ts_dtr*qs_dtr))/nf
    vqq_dtr = (np.sum(vs_dtr*Ts_dtr*qs_dtr))/nf
    wqq_dtr = (np.sum(ws_dtr*Ts_dtr*qs_dtr))/nf
    qqq_dtr = (np.sum(qs_dtr*Ts_dtr*qs_dtr))/nf
    
    uuc_dtr = (np.sum(us_dtr*us_dtr*cs_dtr))/nf
    uvc_dtr = (np.sum(us_dtr*vs_dtr*cs_dtr))/nf
    uwc_dtr = (np.sum(us_dtr*ws_dtr*cs_dtr))/nf
    vvc_dtr = (np.sum(vs_dtr*vs_dtr*cs_dtr))/nf
    vwc_dtr = (np.sum(vs_dtr*ws_dtr*cs_dtr))/nf
    wwc_dtr = (np.sum(ws_dtr*ws_dtr*cs_dtr))/nf
    ucc_dtr = (np.sum(us_dtr*Ts_dtr*cs_dtr))/nf
    vcc_dtr = (np.sum(vs_dtr*Ts_dtr*cs_dtr))/nf
    wcc_dtr = (np.sum(ws_dtr*Ts_dtr*cs_dtr))/nf
    ccc_dtr = (np.sum(cs_dtr*Ts_dtr*cs_dtr))/nf

    #+++++++++++++++++++++++++++++++ Skewness & Kurtosis +++++++++++++++++++++++++++++++++++++++
    # >>> Skewness is a measure of symmetry, or more precisely, the lack of symmetry.  The
    # skewness for a normal distribution is zero, and any symmetric data should have a
    # skewness near zero.  Negative values for the skewness indicate data that are skewed left
    # and positive values for the skewness indicate data that are skewed right.

    # Skewness of the linear detrended data:
    Skew_u = sp.stats.skew(us_dtr)
    Skew_v = sp.stats.skew(vs_dtr)
    Skew_w = sp.stats.skew(ws_dtr)
    Skew_T = sp.stats.skew(Ts_dtr)
    Skew_q = sp.stats.skew(qs_dtr)
    Skew_c = sp.stats.skew(cs_dtr)
    Skew_uw = sp.stats.skew(us_dtr*ws_dtr)
    Skew_vw = sp.stats.skew(vs_dtr*ws_dtr)
    Skew_wT = sp.stats.skew(ws_dtr*Ts_dtr)
    Skew_uT = sp.stats.skew(us_dtr*Ts_dtr)
    Skew_wq = sp.stats.skew(ws_dtr*qs_dtr)
    Skew_uq = sp.stats.skew(us_dtr*qs_dtr)
    Skew_wc = sp.stats.skew(ws_dtr*cs_dtr)
    Skew_uc = sp.stats.skew(us_dtr*cs_dtr)

    # Direct definition of skewness (note that this is the same as 'skewness' MatLab command above):
    Skew_u0=uuu_dtr/sigu_dtr**3
    Skew_v0=vvv_dtr/sigv_dtr**3
    Skew_w0=www_dtr/sigw_dtr**3
    Skew_T0=TTT_dtr/sigT_dtr**3
    Skew_q0=qqq_dtr/sigq_dtr**3
    Skew_c0=ccc_dtr/sigc_dtr**3
    # Use this definition in the case if MatLab Statistics Toolbox is not available

    # >>> Kurtosis is a measure of whether the data are peaked or flat relative to a normal distribution.
    # Data sets with low kurtosis tend to have a flat top near the mean rather than a sharp peak.
    # The kurtosis for a standard normal distribution is three.

    # Kurtosis of the linear detrended data:
    Kurt_u  = sp.stats.kurtosis(us_dtr)
    Kurt_v  = sp.stats.kurtosis(vs_dtr)
    Kurt_w  = sp.stats.kurtosis(ws_dtr)
    Kurt_T  = sp.stats.kurtosis(Ts_dtr)
    Kurt_q  = sp.stats.kurtosis(qs_dtr)
    Kurt_c  = sp.stats.kurtosis(qs_dtr)
    Kurt_uw = sp.stats.kurtosis(us_dtr*ws_dtr)
    Kurt_vw = sp.stats.kurtosis(vs_dtr*ws_dtr)
    Kurt_wT = sp.stats.kurtosis(ws_dtr*Ts_dtr)
    Kurt_uT = sp.stats.kurtosis(us_dtr*Ts_dtr)
    Kurt_wq = sp.stats.kurtosis(ws_dtr*qs_dtr)
    Kurt_uq = sp.stats.kurtosis(us_dtr*qs_dtr)
    Kurt_wc = sp.stats.kurtosis(ws_dtr*cs_dtr)
    Kurt_uc = sp.stats.kurtosis(us_dtr*cs_dtr)

    # Some forth-order moments (for kurtosis calculations):
    uuuu_dtr = (np.sum(us_dtr*us_dtr*us_dtr*us_dtr))/nf
    vvvv_dtr = (np.sum(vs_dtr*vs_dtr*vs_dtr*vs_dtr))/nf
    wwww_dtr = (np.sum(ws_dtr*ws_dtr*ws_dtr*ws_dtr))/nf
    TTTT_dtr = (np.sum(Ts_dtr*Ts_dtr*Ts_dtr*Ts_dtr))/nf
    qqqq_dtr = (np.sum(qs_dtr*qs_dtr*qs_dtr*qs_dtr))/nf
    cccc_dtr = (np.sum(cs_dtr*cs_dtr*cs_dtr*cs_dtr))/nf
    
    # Direct definition of kurtosis (note that this is the same as 'kurtosis' MatLab command above):
    Kurt_u0 = uuuu_dtr/sigu_dtr**4
    Kurt_v0 = vvvv_dtr/sigv_dtr**4
    Kurt_w0 = wwww_dtr/sigw_dtr**4
    Kurt_T0 = TTTT_dtr/sigT_dtr**4
    Kurt_q0 = qqqq_dtr/sigq_dtr**4
    Kurt_c0 = cccc_dtr/sigc_dtr**4

    # Different examples of skewness/kurtosis and PDFs in CBL and SBL can be found in:
    # Chu et al (1996) WRR v32(6) 1681-1688; Graf et al (2010) BLM v134(3) 459-486; Mahrt et al (2012) JPO v42(7) 1134-1142

    # +++++++ Quality Control (QC) parameters derived from sonic anemometer for the 1-hr time series +++++++++++

    # Flagged if the values of different variables exceed user-defined (user-selected) thresholds
    # Angle of attack:
    Phix = phi*180/np.pi

    # If angle of attack is large (say > 15 deg) data should be filtered out or a correction to
    # compensate for the angle of attack error should be applied, e.g. see:

    # van der Molen, M. K. J. H. C. Gash, and J. A. Elbers (2004) Sonic anemometer (co)sine response and flux
    # measurement: II. The effect of introducing an angle of attack dependent calibration. Agricultural and
    # Forest Meterology, 122: 95-109.

    # Nakai, Taro; van der Molen, M. K.; Gash, J. H. C.; Kodama, Yuji (2006) Correction of sonic
    # anemometer angle of attack errors. Agricultural and Forest Meteorology, 136. 19-30.

    # Non-Stationarity of the data in the 1-hr time series, if sigwdir_spc is large (e.g. > 15 deg) data may
    # be considered as "bad" data (non-stationary). However this QC is not applicable for free convection
    # limit (light winds). Another QC parameters: Steadiness of horizontal wind and sonic temperature
    # (non-stationary data)
    Deltau = us_trend[nf-1] - us_trend[0]
    Deltav = vs_trend[nf-1] - vs_trend[0]
    DeltaT = Ts_trend[nf-1] - Ts_trend[0]
    Deltaq = qs_trend[nf-1] - qs_trend[0]
    Deltac = cs_trend[nf-1] - cs_trend[0]
    # For example, |DeltaU| > 2 m/s, |DeltaV| > 2 m/s, |DeltaT| > 2 deg C can be used as QC thresholds

    # Recall that spectral slopes and rotated mean wind components, vrm & wrm, can also be used as QC
    # parameters (see earlier)

    # Different issues of QC are discussed in:

    # Foken, T. and B. Wichura. (1996) Tools for quality assessment of surface-based flux
    # measurements. Agricultural and Forest Meteorology, 78: 83-105.

    # Vickers D., Mahrt L. (1997) Quality control and flux sampling problems for tower and aircraft
    # data. J. Atmos. Oc. Tech. 14(3): 512�526

    # Foken et al. (2004) Edited by X. Lee, et al. Post-field quality control, in Handbook of
    # micrometeorology: A guide for surface flux measurements, 81-108.

    # Burba, G., and D. Anderson, (2010) A Brief Practical Guide to Eddy Covariance Flux Measurements:
    # Principles and Workflow Examples for Scientific and Industrial Applications. LI-COR, Lincoln, USA,
    # Hard and Softbound, 211 pp.

    # if we are missing licor data, make those nan
    if licor_missing == 1:
        wq_csp = nan
        uq_csp = nan
        vq_csp = nan
        Hl     = nan
        CO2_flux = nan
        CO2_flux_Wm2 = nan
        nSq = nan
        nSc = nan
        sqs    = sqs*nan
        cwqs   = cwqs*nan
        cuqs   = cuqs*nan
        cvqs   = cvqs*nan
        Deltaq = Deltaq*nan
        Kurt_q  = Kurt_q*nan
        Kurt_wq = Kurt_wq
        Kurt_uq = Kurt_uq
        Skew_q  = Skew_q*nan
        Skew_wq = Skew_wq*nan
        Skew_uq = Skew_uq*nan
        scs    = sqs*nan
        cwcs   = cwqs*nan
        cucs   = cuqs*nan
        cvcs   = cvqs*nan
        Deltac = Deltaq*nan
        Kurt_c  = Kurt_q*nan
        Kurt_wc = Kurt_wq
        Kurt_uc = Kurt_uq
        Skew_c  = Skew_q*nan
        Skew_wc = Skew_wq*nan
        Skew_uc = Skew_uq*nan

    #
    # End of calculations. Now output something
    #
    turbulence_data = turbulence_data.append([{ \
        'WU_csp': wu_csp,'WV_csp': wv_csp,'UV_csp': uv_csp,'ustar': ustar,'WT_csp': wT_csp,'UT_csp': uT_csp,'VT_csp': vT_csp,'Wq_csp': wq_csp, \
        'Uq_csp': uq_csp,'Vq_csp': vq_csp,'Wc_csp': wc_csp,'Uc_csp': uc_csp,'Vc_csp': vc_csp, \
        'Hs': Hs,'Hl':Hl,'CO2_flux':CO2_flux,'CO2_flux_Wm2':CO2_flux_Wm2, 'Tstar': Tstar,'zeta_level_n': zeta_level_n,'Cd': Cd, \
        'phi_U': phi_u,'phi_V': phi_v,'phi_W': phi_w,'phi_T': phi_T,'phi_UT': phi_uT, \
        'nSU':nSu, 'nSV':nSv, 'nSW':nSw, 'nST':nSt, 'nSq':nSq, 'nSc': nSc, \
        'epsilon_U': epsilon_u,'epsilon_V': epsilon_v,'epsilon_W': epsilon_w,'epsilon': epsilon,'Phi_epsilon': Phi_epsilon, \
        'NT': Nt, \
        'Phi_NT': Phi_Nt, \
        'Phix': Phix, \
        'sigU': urs, 'sigV': vrs, 'sigW': wrs, \
        'DeltaU': Deltau,'DeltaV': Deltav,'DeltaT': DeltaT,'Deltaq': Deltaq,'Deltac': Deltac, \
        'sUs': pd.Series(sus),'sVs':pd.Series(svs),'sWs':pd.Series(sws),'sTs':pd.Series(sTs),'sqs':pd.Series(sqs),'scs':pd.Series(scs), \
        'cWUs':pd.Series(cwus),'cWVs':pd.Series(cwvs),'cWTs':pd.Series(cwTs),'cUTs':pd.Series(cuTs),'cVTs':pd.Series(cvTs),'cWqs':pd.Series(cwqs), \
        'cUqs':pd.Series(cuqs),'cVqs':pd.Series(cvqs),'cWcs':pd.Series(cwcs),'cUcs':pd.Series(cucs),'cVcs':pd.Series(cvcs),'cUVs':pd.Series(cuvs), \
        'fs':pd.Series(fs),'dfs':pd.Series(dfs)}])      

    # # we need to give the columns unique names for the netcdf build later...
    # !! what is the difference betwee dataframe keys and columns? baffled. just change them both.
    # this needs to be done in code to make this more modular

    turbulence_data.keys = turbulence_data.keys()#+'_'+z_level_nominal
    turbulence_data.columns = turbulence_data.keys

    return turbulence_data

# takes datetime object, returns string YYYY-mm-dd
def dstr(date):
    return date.strftime("%Y-%m-%d")

# Bulk fluxes
def cor_ice_A10(bulk_input,le_flag,snow_flag,sta):
# ############################################################################################
# AUTHORS:
#
#   Ola Persson  ola.persson@noaa.gov
#   Python conversion and integration by Christopher Cox (NOAA) christopher.j.cox@noaa.gov
#
# PURPOSE:
#
# Bulk flux calculations for sea ice written by O. Persson and based on the 
# calculations made for SHEBA by Andreas et al. (2004) and the COARE algorithm
# (Fairall et al. 1996) minimization approach to solve for the needed 
# coefficients.
# Python version converted from Matlab cor_ice_A10.m
#
# References:
#   Andreas (1987) https://erdc-library.erdc.dren.mil/jspui/bitstream/11681/9435/1/CR-86-9.pdf
#   Andreas et al. (2004) https://ams.confex.com/ams/7POLAR/techprogram/paper_60666.htm
#   Andreas et al. (2004) https://doi.org/10.1175/1525-7541(2004)005<0611:SOSIAN>2.0.CO;2
#   Fairall et al. (1996) https://doi.org/10.1029/95JC03205                                 
#   Fairall et al. (2003) https://doi.org/10.1175/1520-0442(2003)016<0571:BPOASF>2.0.CO;2
#   Holtslag and De Bruin (1988) https://doi.org/10.1175/1520-0450(1988)027<0689:AMOTNS>2.0.CO;2
#   Grachev and Fairall (1997) https://doi.org/10.1175/1520-0450(1997)036<0406:DOTMOS>2.0.CO;2
#   Paulson (1970) https://doi.org/10.1175/1520-0450(1970)009<0857:TMROWS>2.0.CO;2
#   Smith (1988) https://doi.org/10.1029/JC093iC12p15467
#   Webb et al. (1980) https://doi.org/10.1002/qj.49710644707
# 
# Notes:
#   - prior to mods (see updates): x=[4.5,0,-10,-5,1,2,203,250,0,600,1010,15,15,15] test data compares to matlab version < 10^-15 error (Cox)
#   - modified for ice or water (Persson)
#   - this version has a shortened iteration (Persson) [max iters in stable conditions = 3]
#   - the 5th variable is a switch, x(5)<3 means ice, >3 means water (Persson)
#   - uses LKB Rt and Rq for seawater and Andreas 1987 for ice (Persson)
#   - presently has fixed zo=3e-4 for ice and Smith (1988) for water (Persson)    
#   - First guess Qs from Buck (qice.m, qwat.m by O. Persson) but hard-coded here. These estimates will
#     differ slightly from Hyland and Wexler humidity calculations reported at the tower (Cox)
#  
# Updates:
#   - additional documentation (Cox)
#   - instead of passing LWD, SWD and estimating nets, code now expects netLW and netSW (Cox)
#   - included rr, rt, rq as outputs
#   - took nominal zot and zoq out of the loop
#   - calculating zot, zoq and zo inside the loop now, but not allowing zoq or zot to be smaller than nominal 10^-4 Andreas value
#   - removed rain rate
#   - removed cool skin and hardcoded iceconcentration to be 1
# 
# Outputs:
#   hsb: sensible heat flux (Wm-2)
#   hlb: latent heat flux (Wm-2)
#   tau: stress                             (Pa)
#   zo: roughness length, veolicity              (m)
#   zot:roughness length, temperature (m)
#   zoq: roughness length, humidity (m)
#   L: Obukhov length (m)
#   usr: friction velocity (sqrt(momentum flux)), ustar (m/s)
#   tsr: temperature scale, tstar (K)
#   qsr: specific humidity scale, qstar (kg/kg?)
#   dter:
#   dqer: 
#   hl_webb: Webb density-corrected Hl (Wm-2)
#   Cd: transfer coefficient for stress
#   Ch: transfer coefficient for Hs
#   Ce: transfer coefficient for Hl
#   Cdn_10: 10 m neutral transfer coefficient for stress
#   Chn_10: 10 m neutral transfer coefficient for Hs
#   Cen_10: 10 m neutral transfer coefficient for Hl
#   rr: Reynolds number
#   rt: 
#   rq:
    

    import math
     
    u=bulk_input[0]         # wind speed                         (m/s)
    ts=bulk_input[1]        # bulk water/ice surface tempetature (degC)
    t=bulk_input[2]         # air temperature                    (degC) 
    Q=bulk_input[3]         # air moisture mixing ratio          (kg/kg)
    zi=bulk_input[4]        # inversion height                   (m)
    P=bulk_input[5]         # surface pressure                   (mb)
    zu=bulk_input[6]        # height of anemometer               (m)
    zt=bulk_input[7]        # height of thermometer              (m)
    zq=bulk_input[8]        # height of hygrometer               (m)
    rh=bulk_input[9]        # relative humidity                  (unitless)
    vwc=bulk_input[10]      # volume water content               (m3/m3)
    
    
    ################################# Constants ################################## 
    
    # mixing ratio to specific humidity
    Q = Q/(1+Q)
    
    # Set
    Beta=1.25 # gustiness coeff
    von=0.4 # von Karman constant
    fdg=1.00 # ratio of thermal to wind von Karman
    tdk=273.15 
    grav=9.82 # gravity

    # Air
    Rgas=287.1
    cpa=1004.67 
    rhoa=P*100/(Rgas*(t+tdk)*(1+1.61*Q)) # density of air
    visa=1.325e-5*(1+6.542e-3*t+8.301e-6*t*t-4.8e-9*t*t*t) # kinematic viscosity
    
    ##############################################################################
    
    
    ############################### Subfunctions ################################# 
    
    # psi_m and psi_h are the stability functions that correct the neutral 
    # stability calculations for drag and tranfer, respectively, for deviations
    # from that assumption. They are "known" parameters and are borrowed here
    # from Paulson (1970) for the unstable state and Holtslag and De Bruin 
    # (1988) for the stable case, as wad done for SHEBA (Andreas et al. 2004).
    
    # for drag
    def psih_sheba(zet):
        if zet<0: # instability, Paulson (1970)
            x=(1-15*zet)**.5
            psik=2*math.log((1+x)/2)
            x=(1-34.15*zet)**.3333
            psic=1.5*math.log((1+x+x*x)/3)-math.sqrt(3)*math.atan((1+2*x)/math.sqrt(3))+4*math.atan(1)/math.sqrt(3)
            f=zet*zet/(1+zet*zet)
            psi=(1-f)*psik+f*psic                                              
        else: # stability, Holtslag and De Bruin (1988)
            ah = 5 
            bh = 5 
            ch = 3
            BH = math.sqrt(ch**2 - 4)
            psi = - (bh/2)*math.log(1+ch*zet+zet**2) + (((bh*ch)/(2*BH)) - (ah/BH))*(math.log((2*zet+ch-BH)/(2*zet+ch+BH)) - math.log((ch-BH)/(ch+BH)))
        return psi
    
    # for transfer
    def psim_sheba(zet):
        if zet<0: # instability, Paulson (1970)
            x=(1-15*zet)**.25;
            psik=2*math.log((1+x)/2)+math.log((1+x*x)/2)-2*math.atan(x)+2*math.atan(1)
            x=(1-10.15*zet)**.333
            psic=1.5*math.log((1+x+x*x)/3)-math.sqrt(3)*math.atan((1+2*x)/math.sqrt(3))+4*math.atan(1)/math.sqrt(3)
            f=zet*zet/(1+zet*zet)
            psi=(1-f)*psik+f*psic                                             
        else: # stability, Holtslag and De Bruin (1988)
            am = 5 
            bm = am/6.5
            BM = ((1-bm)/bm)**(1/3)
            y = (1+zet)**(1/3)
            psi = - (3*am/bm)*(y-1)+((am*BM)/(2*bm))*(2*math.log((BM+y)/(BM+1))-math.log((BM**2-BM*y+y**2)/(BM**2-BM+1))+2*math.sqrt(3)*math.atan((2*y-BM)/(BM*math.sqrt(3)))-2*math.sqrt(3)*math.atan((2-BM)/(BM*math.sqrt(3))))
        return psi
    
    

    ########################### COARE BULK LOOP ##############################
         
    # First guesses 
    
    if le_flag == 1: # snow cover
        Le=(2.501-.00237*ts)*1e6 + 333.55*1000 # [J/kg] latent heat of sublimation
    elif le_flag == 0: # snow free or melting surface
        Le=(2.501-.00237*ts)*1e6 # [J/kg] latent heat of vaporization 

    if snow_flag == 1: # snow cover 
        es=(1.0003+4.18e-6*P)*6.1115*math.exp(22.452*ts/(ts+272.55)) # saturation vapor pressure
        Qs=es*622/(P-.378*es)/1000 # Specific humidity (Bolton, 1980)    
        dq=Qs-Q
        if sta == 'asfs30':
            zogs=2.3e-4 # Andreas et al. winter snow
        elif sta == 'asfs50':
            zogs=2.3e-4 # Andreas et al. winter snow
    elif snow_flag == 0: # snow free   
        es=6.112*math.exp(17.502*ts/(ts+241.0))*(1.0007+3.46e-6*P) # saturation vapor pressure
        Qs=es*622/(P-.378*es)/1000 # Specific humidity (Bolton, 1980)  
        Beta = 0.25*(1-np.cos(math.pi*(vwc/0.35)))**2 # Lee and Pielke (1991)
        if vwc > 0.35: Beta = 1 # Lee and Pielke (1991)
        Alpha = rh + Beta*(1-rh) # Kondo et al. (1990)
        dq=(Alpha*Qs-Q)
        if sta == 'asfs30':
            zogs=3.6e-3 # Mean z0 from Jun/Jul/Aug/Sep/Oct
        elif sta == 'asfs50':
            zogs=2.4e-3 # Mean z0 from Jun/Jul/Aug/Sep/Oct
        
    
          
    wetc=0.622*Le*Qs/(Rgas*(ts+tdk)**2)
            
    du=u
    dt=ts-t-0.0098*zt
    ta=t+tdk
    ug=0.5
    dter=0
    
    ut=math.sqrt(du*du+ug*ug)    
    
    # Neutral coefficient
    u10=ut*math.log(10/zogs)/math.log(zu/zogs)
    cdhg=von/math.log(10/zogs)    
    usr=cdhg*u10
    zo10=zogs
    
    Cd10=(von/math.log(10/zo10))**2
    Ch10=0.0015
    
    Ct10=Ch10/math.sqrt(Cd10)
    zot10=10/math.exp(von/Ct10)
    Cd=(von/math.log(zu/zo10))**2

    # Grachev and Fairall (1997)    
    Ct=von/math.log(zt/zot10) # temp transfer coeff
    CC=von*Ct/Cd # z/L vs Rib linear coefficient
    Ribcu=-zu/zi/.004/Beta**3 # Saturation Rib
    Ribu=-grav*zu/ta*((dt-dter)+.61*ta*dq)/ut**2
    nits=7 # Number of iterations.
    
    if Ribu<0:
    	zetu=CC*Ribu/(1+Ribu/Ribcu) # Unstable, G&F
    else:
    	zetu=CC*Ribu*(1+27/9*Ribu/CC) # Stable, I forget where I got this	
        
    L10=zu/zetu # MO length
    
    if zetu>150:
     	nits=3 # cutoff iteration if too stable
    
    # Figure guess stability dependent scaling params 
    usr=ut*von/(math.log(zu/zo10)-psim_sheba(zu/L10))
    tsr=-(dt-dter)*von*fdg/(math.log(zt/zot10)-psih_sheba(zt/L10))
    qsr=-(dq-wetc*dter)*von*fdg/(math.log(zq/zot10)-psih_sheba(zq/L10))
  
    zot=1e-4
    zoq=1e-4 # approximate values found by Andreas et al. (2004)  		    
    
    # Bulk Loop
    for i in range(nits): 
     

        zet=von*grav*zu/ta*(tsr+0.61*ta*qsr)/(usr**2) # Eq. (7), Fairall et al. (1996)
        zo=zogs
        
        # Fairall et al. (2003)
        rr=zo*usr/visa # Reynolds
        
        # Andreas (1987) for snow/ice
        if rr<=0.135: # areodynamically smooth
            rt=rr*math.exp(1.250)
            rq=rr*math.exp(1.610)
        elif rr<=2.5: # transition
            rt=rr*math.exp(0.149-.55*math.log(rr))
            rq=rr*math.exp(0.351-0.628*math.log(rr))
        else: # aerodynamically rough
            rt=rr*math.exp(0.317-0.565*math.log(rr)-0.183*math.log(rr)*math.log(rr))
            rq=rr*math.exp(0.396-0.512*math.log(rr)-0.180*math.log(rr)*math.log(rr))
    

        L=zu/zet
        usr=ut*von/(math.log(zu/zo)-psim_sheba(zu/L))
        tsr=-(dt-dter)*von*fdg/(math.log(zt/zot)-psih_sheba(zt/L))
        qsr=-(dq-wetc*dter)*von*fdg/(math.log(zq/zoq)-psih_sheba(zq/L))
        Bf=-grav/ta*usr*(tsr+0.61*ta*qsr)
       
        if Bf>0:
            ug=Beta*(Bf*zi)**0.333
        else:
            ug=0.2
            
                
        # #########    
        # # added by Cox...
        # # this allows zo to be calculated in the loop, which I think it needs to be
        # # zot as well, but not allowing it to go to 0 and brek things.
        # tau=rhoa*usr*usr*du/ut # stress
        # Cd=tau/rhoa/du**2
        # Ch=-usr*tsr/du/(dt-dter)
        # zogs=zu*math.exp( -(von*Cd**-0.5 + psim_sheba(zq/L)) ) # Andreas (2004) eq. 3A
        # zot=zu* math.exp( -(von*Cd**-0.5*Ch**-1 + psih_sheba(zq/L)) ) # Andreas (2004) eq. 3B
        # if zot < 1e-4: zot = 1e-4
        # ##########
        
        
        ut=math.sqrt(du*du+ug*ug)
        hsb=-rhoa*cpa*usr*tsr
        hlb=-rhoa*Le*usr*qsr
    
        dter=0
       
        dqer=wetc*dter
        

    # end bulk iter loop
    
    tau=rhoa*usr*usr*du/ut # stress

    ##############################################################################

    # Webb et al. correction following Fairall et al 1996 Eqs. 21 and 22
    wbar=1.61*(hlb/rhoa/Le)+(1+1.61*Q)*(hsb/rhoa/cpa)/ta
    hl_webb=hlb+(rhoa*Le*wbar*Q)
    
    # compute transfer coeffs relative to du @meas. ht
    Cd=tau/rhoa/du**2
    Ch=-usr*tsr/du/(dt-dter)
    Ce=-usr*qsr/(dq-dqer)/du
    # 10-m neutral coeff realtive to ut
    Cdn_10=von**2/math.log(10/zo)/math.log(10/zo)
    Chn_10=von**2*fdg/math.log(10/zo)/math.log(10/zot)
    Cen_10=von**2*fdg/math.log(10/zo)/math.log(10/zoq)
    
    bulk_return=[hsb,hl_webb,tau,zo,zot,zoq,L,usr,tsr,qsr,dter,dqer,hlb,Cd,Ch,Ce,Cdn_10,Chn_10,Cen_10,rr,rt,rq]
    
    return bulk_return



# QCRAD
def qcrad(df,sw_range,lw_range,D1,D5,D11,D12,D13,D14,D15,D16,A0):

    # Screen for outliers in the radiation data using select tests from
    # Long and Shi (2008) https://doi.org/10.2174/1874282300802010023
    #
    # The following tests are performed:
    #   Shortwave
    #       - PPL
    #       - Climatological Configurable (global) Limits (SWD, SWU)
    #       - Rayleigh Limit (SWD)
    #       - Albedo (failure removes both; not structly QCRAD and not very strict. Just a sanity check)
    #   Longwave
    #       - PPL
    #       - Climatological Configurable Tair vs LW (LWD, LWU)
    #       - Climatological Configurable LWD vs LWU (LWD

    # Constants and Setup
    mu0 = np.cos(np.deg2rad(df['zenith_true']))
    mu0.loc[mu0 < 0] = 0
    au = 1
    Sa  = 1368/au**2
    sb  = 5.67*1e-8
        
    #
    # Shortwave
    #
    
    # (1) PPL
    df['down_short_hemisp'].mask( (df['down_short_hemisp']<sw_range[0]) | (df['down_short_hemisp']>sw_range[1]) , inplace=True)
    df['up_short_hemisp'].mask( (df['up_short_hemisp']<sw_range[0]) | (df['up_short_hemisp']>sw_range[1]) , inplace=True)

    # (2) Rayleigh Limit
    # Technically it also needs a diffuse qualifier, but we don't have that available.
    # Howevr, this limit is unlikely to be met at the SZA range for MOSAiC because of the 50 Wm2 qualifier anyhow so this is largely a formality. 
    RL = 209.3*mu0 - 708.3*mu0**2 + 1128.7*mu0**3 - 911.2*mu0**4 + 287.85*mu0**5 + 0.046725*mu0*df['atmos_pressure'].mean()   
    df['down_short_hemisp'].mask( (df['down_short_hemisp'] < RL-1.0) & (df['down_short_hemisp'] > 50), inplace=True)
    
    # (3) CCL
    ccswdH2 = Sa*D1*mu0**1.2 + 55
    ccswuH2 = Sa*D5*mu0**1.2 + 55
    df['down_short_hemisp'].mask( df['down_short_hemisp'] > ccswdH2, inplace=True)
    df['up_short_hemisp'].mask( df['up_short_hemisp'] > ccswuH2, inplace=True)
     
    # (4) Albedo
    albs_abs = np.abs(df['up_short_hemisp']/df['down_short_hemisp'])
    df['down_short_hemisp'].mask( (albs_abs > A0) & (mu0 > np.cos(np.deg2rad(80))), inplace=True) 
    df['up_short_hemisp'].mask( (albs_abs > A0) & (mu0 > np.cos(np.deg2rad(80))), inplace=True) 


    #
    # Longwave
    #
     
    # (1) PPL
    df['down_long_hemisp'].mask( (df['down_long_hemisp']<lw_range[0]) | (df['down_long_hemisp']>lw_range[1]) , inplace=True)
    df['up_long_hemisp'].mask( (df['up_long_hemisp']<lw_range[0]) | (df['up_long_hemisp']>lw_range[1]) , inplace=True) 
     
    # (2) Tair
    Tair = (df['temp']+273.15).interpolate()
    tair_lwd_lo = D11 * sb*(Tair)**4 
    tair_lwd_hi = sb*Tair**4 + D12
    tair_lwu_lo = sb*(Tair-D13)**4
    tair_lwu_hi = sb*(Tair+D14)**4
    df['down_long_hemisp'].mask( (df['down_long_hemisp']<tair_lwd_lo) | (df['down_long_hemisp']>tair_lwd_hi) , inplace=True)
    df['up_long_hemisp'].mask( (df['up_long_hemisp']<tair_lwu_lo) | (df['up_long_hemisp']>tair_lwu_hi) , inplace=True)
    
    # (3) LWU <-> LWD 
    lwd_lwu_lo = df['up_long_hemisp'] - D15
    lwd_lwu_hi = df['up_long_hemisp'] + D16
    df['down_long_hemisp'].mask( (df['down_long_hemisp']<lwd_lwu_lo) | (df['down_long_hemisp']>lwd_lwu_hi) , inplace=True)
    
    
    # return screened values
    return df
 


# Tilt correction
def tilt_corr(df,diff):

    # Performs a pyranometer tilt correction using a method developed from Long et al. (2010)
    # called from create_level2_product_asfs.py
    #
    # df: the dataframe for this day (sdt in create_level2_product_asfs.py)
    # diff: dataframe having the same index as df containing measured diffuse flux. if -1 signals no measuremnt available and the code will parameterize diffuse flux instead
    # incx_offset: value of metek incX at last installation; passed through df
    # incy_offset: value of metek incY at last installation; passed through df

    ###### (0) For documentation we will do this in the parlance of Long et al. (2010)
    G_t = df['down_short_hemisp'] # observed SWD in the tilted frame 
    
    ###### (1) Calculate clear-sky equivalent SWD
    #   Based on a power-law relationship following Long and Shi (2008) which is the lite-version of Long and Ackerman (2000)
    a_coef = 1266 # Wm^-2
    b_coef = 1.1740 
    mu0 = np.cos(np.deg2rad(df['zenith_true']))
    mu0.loc[mu0 < 0] = 0
    G_clr = a_coef*mu0**b_coef # clear-sky flux
    
    ###### (2) Get your diffuse, either by parameterizing or measurement passed as argument
    # First Parameterize
    # (i) Calcualte Rayleigh Limit
    #   Using STREAMER 3.1, arctic summer stdatm, no aerosols, and albedo scaled such that sza=70 is albedo = 0.8
    diff_calc_891_noaerosol  = [131.94,131.54,130.29,128.21,125.32,121.65,117.23,112.09,106.27, 99.82, 92.77,85.14,76.96,68.21,58.83,48.63,37.13,22.99,0.8]
    diff_calc_1010_noaerosol = [146.39,145.95,144.52,142.17,138.90,134.74,129.74,123.92,117.35,110.05,102.07,93.46,84.23,74.37,63.82,52.39,39.60,24.09,0.8]
    RL = np.polyval(np.polyfit(np.cos(np.deg2rad(np.linspace(0,90,19))),diff_calc_1010_noaerosol,4),mu0)
    RL[mu0 < 0] = 0
    RL = pd.DataFrame(data=RL,columns=['RL'],index=df.index)
    
    # (ii) Define a variable, sky transmissivity
    #   Only use data where the SZA < 88 degrees: when the sun is below this, no correction will be made
    trans = G_t/G_clr

    # (iii) Parameterize the diffuse fraction
    #
    # (a) define a variable, diffuse fraction
    diff_fract = np.nan*G_t
    #
    # (b) identify the clear skies (very) loosely based on Long and Ackerman (2000) by thresholding sky transmissivity and time variance
    smoothed_cre = (G_t-G_clr).interpolate(method='pad').rolling(11,min_periods=1,center=True).mean() 
    ind_clr = (trans > 0.7) & (abs(smoothed_cre) < 100) | (G_t > G_clr)
    diff_fract.loc[ind_clr] = RL['RL'].where(ind_clr)/G_t.where(ind_clr)
    #
    # (c) identify the cloudy skies and use a parameterization based on D-ICE
    ind_cld = (ind_clr==False) & (~np.isnan(trans)) & (df['zenith_true'] < 89.9)
    diff_fract.loc[ind_cld] = (0.94958*G_t.where(ind_cld) - 5.8128)/G_t.where(ind_cld)

    D_param =  G_t*diff_fract # diffuse flux

    # Now, if a measurement is available use it and fill gaps with the parameterization
    if isinstance(diff,pd.Series): # if we did not observe diffuse
        D = diff # formality. this is the observed diff
        D = D_param.mask(D>-9000,D) # if there are any missing values, use the paramerterization    
    else:       
        # (iv) estimate D
        D = D_param # diffuse flux
        
    ###### (3) from D, calculate N and D_t
    D_t = D # Diffuse at the tilted sensor. formality. We set tilted diffuse to measured (parameterized) diffuse because as Chuck says "diffuse is aptly named"
    N = (G_t-D_t)/mu0 # Direct normal to sun's rays
    
    ###### (4) Calculate slope and aspect of the G_t
    
    # we cannot know the aspect of the sr30 precisely so we use the metek but the metek was installed with its own tilt so we remove that here
    # we assume that at installation the level was 0 (even if it wasn't) and we will correct for all CHANGE in the tilt from that moment
    # we also assume that for the affected period the RELATIVE orientation of sr30 and metek was a constant  
    incX = df['metek_InclX_Avg'] - df['incx_offset']
    incY = df['metek_InclY_Avg'] - df['incy_offset']

    # this is the aspect of the unlevel plane of the thermopile
    aspect = np.mod(df['heading'] - np.rad2deg(np.arctan(np.cos(np.deg2rad(incY))/np.cos(np.deg2rad(incX)))),360)
    
    # this is the slope of the unlevel plane of the thermopile
    slope = np.sqrt(incX**2 + incY**2)
    
    ###### (5) Calcualte mu_t, the solar zenith angle relative to the unlevel plane of the thermopile
    mu_t = np.cos(np.deg2rad(df['zenith_true']))*np.cos(np.deg2rad(slope))+np.sin(np.deg2rad(df['zenith_true']))*np.sin(np.deg2rad(slope))*np.cos(np.deg2rad(df['azimuth'])-np.deg2rad(aspect)) # Eq. 1.6.3 from Duffie & Beckman (2013)

    ###### (6) Implement the correction following Long et al. (2010)
    G = G_t * ((mu0 + D/N) / (mu_t + D/N))
        
    ###### (7) and insert the result back into the df then return it
    G[~G.notnull()] = df['down_short_hemisp'] # where G is nan replace with original measurement. this happens when there is no correction to make
    df['down_short_hemisp'] = G
    
    return df
 
# takes in any array and calls numpy interpolation instead of pandas
def interpolate_nans_vectorized(arr):
    
    nans, x =  np.isnan(arr), lambda z: z.nonzero()[0]
    try: arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
    except ValueError:
        arr = arr*nan
    return arr

# takes in a pandas series of qc flags and returns the 10 minute "average" of those
# flags, according to this defined logic
def average_flags(qc_series, fstr):

    # Implement/fix the data averaging and qc flags into 10-min space (i.e., no averaging of bad data).  If
    # >= 5 good then average good + flag good; If >= 5 caution or good then average those + flag caution; If
    # >= 5 engineering then average those + flag engineering; Otherwise, average all + flag bad
    # 0 = good, 1 = caution, 2 = bad, 3 = engineering
    def take_qc_average(data_series):
        tot_good             = (data_series == 0).sum()
        tot_caution_and_good = tot_good + (data_series == 1).sum()
        tot_engineering      = (data_series == 3).sum()

        #print(f"{tot_good} -- {tot_caution_and_good} -- {tot_engineering}")
        if tot_good/len(data_series) >= 0.5:             return 0 # if more than half are good, we're good
        if tot_caution_and_good/len(data_series) >= 0.5: return 1 # if more than half are caution&&good, we're caution
        if tot_engineering/len(data_series) >= 0.5:      return 3 # if more than half are engineering, we're engineers

        # this should only affect data from turbulence calculations, are there an other places that
        # would have an issue with this ugly fix??
        if data_series.isna().sum() == len(data_series): return nan

        return 2                                                  # any other combination is bad bad not good

    return qc_series.resample(fstr, label='left').apply(take_qc_average)


# little helper function that will write your variables to a pickle file for debugging function
# complex/deeply nested calls
def pickle_function_args_for_debugging(args, pickle_dir='./', pickle_name='tmp_file.pkl'):
    import pickle
    from datetime import date
    pkl_file = open(f'{pickle_dir}/{date.today().strftime("%Y%m%d")}_{pickle_name}', 'wb')
    pickle.dump(args, pkl_file)
    pkl_file.close()
