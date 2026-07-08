import numpy as np
import pvlib
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt

# Use np.trapezoid if available (NumPy >= 2.0), otherwise fall back to np.trapz
trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


class Surface(object):
    """
    The surface class contains all relevant information about the surface material and position.
    """
    def __init__(self,
                 solar_reflectivity,
                 spectral_absorptivity,
                 tilt,
                 azimuth,
                 sky_view_factor=None,
                 surrounds_albedo=0.15,
                 name=None,
                 latitude=-33.8688,
                 longitude=151.2093,
                 thickness=8e-4,
                 spec_heat_cap=890,
                 density=2710,
                 convection_model='Maestre-Valero',
                 char_length=None):

        self.solar_reflectivity = solar_reflectivity
        self.spectral_absorptivity = spectral_absorptivity
        self.tilt = tilt
        self.azimuth = azimuth
        self.name = name
        self.surrounds_albedo = surrounds_albedo
        self.latitude = latitude
        self.longitude = longitude
        self.thickness = thickness
        self.spec_heat_cap = spec_heat_cap
        self.density = density
        self._sky_view_factor = sky_view_factor
        self.convection_model = convection_model
        self.char_length = char_length

    @property
    def sky_view_factor(self):
        if self._sky_view_factor is not None:
            return self._sky_view_factor
        else:
            return np.cos(np.deg2rad(self.tilt))

    @property
    def heat_cap_per_m2(self):
        return self.spec_heat_cap*self.thickness*self.density


class SurfaceSimulator(object):
    def __init__(self,
                 surface,
                 prad_function,
                 exp_df
                 ):
        self.surface = surface
        self.Prad_function = prad_function
        self.Prad_function.surface = surface
        self.exp_df = exp_df

        self.interp_Tamb = self.create_interpolator('Tamb')
        self.interp_Humid = self.create_interpolator('Humidity')
        self.interp_wspd = self.create_interpolator('Windspeed')
        self.interp_pres = self.create_interpolator('Barometric pressure, kPa')
        self.interp_Psolar = self.create_interpolator('irradiance on tilted panels')
        self.interp_cloud = self.create_interpolator('Cloud cover')

        self.heatcap_water = 4184

    def create_interpolator(self, name):
        return interp1d(self.exp_df['Time, s'], self.exp_df[name], kind='linear', bounds_error=False, fill_value=(self.exp_df[name].iloc[0], self.exp_df[name].iloc[-1]))

    def _derivs(self, t, Y):
        # Physically, condensed mass cannot be negative
        condensed_mass = max(0.0, Y[0])
        Tsurf = Y[1]
        c = Y[2]

        Tamb = self.interp_Tamb(t)
        pressure = self.interp_pres(t)
        windspeed = self.interp_wspd(t)
        humidity = self.interp_Humid(t)
        irrad = self.interp_Psolar(t)

        heat_cap_surf = 1/(self.surface.heat_cap_per_m2+condensed_mass*self.heatcap_water)
        cond_rate = deriv_m(Tamb, humidity, Tsurf, windspeed, pressure, tilt=self.surface.tilt, model=self.surface.convection_model, char_length=self.surface.char_length)
        dmdt = cond_rate-0.01*condensed_mass #I think this is just hacking in an evaporation rate?
        dcdt = 0.01*condensed_mass # c represents cumulative evaporated mass

        # Clamp derivatives if we reach 0 condensed mass and evaporation exceeds condensation
        if condensed_mass == 0.0 and dmdt < 0:
            dmdt = 0.0
            dcdt = 0.0

        dTdt =  heat_cap_surf*heat_balance_equation(Tsurf_guess=[Tsurf], Tamb=Tamb, irradiance=irrad, humidity=humidity, windspeed=windspeed,
                                                           reflectivity=self.surface.solar_reflectivity, radht=self.Prad_function, cloud_cover=self.interp_cloud(t),
                                                           cond_rate=cond_rate, model=self.surface.convection_model, char_length=self.surface.char_length)
        return [dmdt,dTdt,dcdt]

    def solve(self):
        return solve_ivp(self._derivs,
                          [0,self.exp_df['Time, s'].iloc[-1]],
                          [0,self.interp_Tamb(0), 0])

    def run_simulation(self):
        """
        Runs the solver and interpolates results back onto the experimental time grid,
        returning a populated DataFrame directly compatible with plotting.
        """
        sol = self.solve()
        t_exp = self.exp_df['Time, s'].values

        interp_mass = interp1d(sol.t, sol.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_Tsurf = interp1d(sol.t, sol.y[1], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_c = interp1d(sol.t, sol.y[2], kind='linear', bounds_error=False, fill_value="extrapolate")

        sim_df = self.exp_df.copy()
        sim_df['Theory - condensed_mass'] = np.maximum(0.0, interp_mass(t_exp))
        sim_df['Theory - Tsurf'] = interp_Tsurf(t_exp)
        sim_df['Theory - c'] = interp_c(t_exp)

        # Calculate condensation rate
        cond_rates = []
        for t, Tsurf in zip(t_exp, sim_df['Theory - Tsurf']):
            Tamb = self.interp_Tamb(t)
            pressure = self.interp_pres(t)
            windspeed = self.interp_wspd(t)
            humidity = self.interp_Humid(t)
            cond_rates.append(deriv_m(Tamb, humidity, Tsurf, windspeed, pressure, tilt=self.surface.tilt, model=self.surface.convection_model, char_length=self.surface.char_length))
        sim_df['Theory - cond rate'] = np.array(cond_rates)

        # Cumulative condensation is the current condensed mass plus cumulative evaporated mass
        sim_df['Theory - cumulative condensation'] = sim_df['Theory - condensed_mass'] + sim_df['Theory - c']

        return sim_df

def calculate_plane_of_array_irradiance(surface, times, exp_ghi):
    # Would be nice to have a way to account for the loss of dhi due to surrounding buildings etc
    # raising the horizon.
    solar_pos = pvlib.solarposition.get_solarposition(time=times, latitude=surface.latitude, longitude=surface.longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    erbs = pvlib.irradiance.erbs(exp_ghi, solar_pos['zenith'], times)

    return pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface.tilt,
            surface_azimuth=surface.azimuth, # south
            solar_zenith=solar_pos['apparent_zenith'],
            solar_azimuth=solar_pos['azimuth'],
            dni=erbs['dni'],
            ghi=exp_ghi,
            dhi=erbs['dhi'],
            dni_extra=dni_extra,
            model='perez-driesse',
            albedo=surface.surrounds_albedo
            )['poa_global']


class equil_solver(object):
    def __init__(self, surface, radht, Tamb, humidity=50, pressure=101.325, windspeed=0, cloud_cover=0):
        self.surface = surface
        self.radht = radht
        self.humidity = humidity
        self.pressure = pressure
        self.windspeed = windspeed
        self.cloud_cover = cloud_cover
        self.Tamb = Tamb

    def T_ghi(self, ghi, reflectivity):
        self.surface.solar_reflectivity = reflectivity
        # self.surface.spectral_absorptivity.LWIRemm = LWIRemm
        self.surface.spectral_absorptivity.SWIRemm = 1-reflectivity
        self.radht.update_surface_cache()

        ghi_basis = np.linspace(0,1100,11)
        Tsurfs = []
        initial_guess = [self.Tamb]

        # Step 2: Use fsolve to find the exact Tc that makes the heat_balance_equation return 0.
        # We pass all the environmental variables through the 'args' tuple.
        for i_ghi in ghi_basis:
            Tc_equilibrium, = fsolve(
                heat_balance_equation,
                x0=initial_guess,
                args=(self.Tamb, i_ghi, self.surface.solar_reflectivity, self.radht, self.humidity, self.windspeed, self.cloud_cover, 0.0, self.surface.convection_model, self.surface.char_length)
            )
            Tsurfs.append(Tc_equilibrium)
        GHI_interp = interp1d(ghi_basis, np.array(Tsurfs), kind='linear')

        return GHI_interp(ghi) - self.Tamb


def make_dataframe(df_section, surface):
    """
    Somewhat dodgy: the calculations for the solar position get done in here
    """
    new_df = pd.DataFrame(index=df_section.index)
    new_df['Time, s'] = (df_section.index - df_section.index[0]).total_seconds()
    new_df['Barometric pressure, kPa'] = 0.1*df_section[('General', 'Barometric pressure, mbar')]
    new_df['Tsurf'] = (df_section[(surface.name, 'Temp 1')] + df_section[(surface.name, 'Temp 2')])/2
    new_df['Tamb'] = df_section[('General', 'Air temp, ˚C')]
    new_df['Humidity'] = df_section[('General', 'RH, %')]
    new_df['Windspeed'] = df_section[('General', 'Windspeed, m/s')]
    new_df['Cloud cover'] = df_section[('ERA5_Weather', 'Total cloud cover, fraction')]
    new_df['GHI'] = df_section[('General', 'Solar flux density, W/m^2')]
    new_df['irradiance on tilted panels'] = calculate_plane_of_array_irradiance(surface=surface,
                                                                                 times=df_section.index,
                                                                                 exp_ghi=df_section[('General', 'Solar flux density, W/m^2')])
    
    dew_col = (surface.name, 'Dew collected, mL/m^2')
    if dew_col in df_section.columns:
        new_df['Dew collected, mL/m2'] = df_section[dew_col]
    else:
        new_df['Dew collected, mL/m2'] = 0.0

    return new_df

## -----------
def get_tarp_natural_convection(T_surface, T_ambient, tilt, char_length=None):
    """
    Computes natural convection heat transfer coefficient.
    If char_length is provided, uses size-dependent McAdams/ASHRAE correlations.
    Otherwise, defaults to the size-independent George N. Walton (1983) model.
    T_surface and T_ambient are in Celsius, tilt is in degrees, char_length in meters.
    """
    delta_T = T_surface - T_ambient
    abs_delta_T = abs(delta_T)
    if abs_delta_T < 1e-5:
        return 0.0
    
    theta = np.deg2rad(tilt)
    cos_theta = np.cos(theta)
    abs_cos_theta = abs(cos_theta)
    
    is_upward = (tilt <= 90.0)
    is_heating = (delta_T > 0.0)
    is_unstable = (is_upward and is_heating) or (not is_upward and not is_heating)
    
    if char_length is not None and char_length > 0.0:
        # Size-dependent natural convection based on Rayleigh number
        # Ra_L = g * beta * delta_T * L^3 / (nu * alpha)
        # Approximate scaling factor for air at standard temperature: 1.6e8
        Ra_L = 1.6e8 * abs_delta_T * (char_length ** 3)
        
        # Horizontal plate component
        if is_unstable:
            if Ra_L < 1e7:
                h_nat_horiz = 1.32 * ((abs_delta_T / char_length) ** 0.25)
            else:
                h_nat_horiz = 1.52 * (abs_delta_T ** (1/3))
        else:
            h_nat_horiz = 0.59 * ((abs_delta_T / char_length) ** 0.25)
            
        # Vertical plate component
        if Ra_L < 1e9:
            h_nat_vert = 1.42 * ((abs_delta_T / char_length) ** 0.25)
        else:
            h_nat_vert = 1.31 * (abs_delta_T ** (1/3))
            
        # Interpolate based on tilt angle
        sin_theta = np.sin(theta)
        h_nat = h_nat_horiz * abs_cos_theta + h_nat_vert * abs(sin_theta)
    else:
        # Size-independent Walton model
        if is_unstable:
            h_nat = 9.482 * (abs_delta_T ** (1/3)) / (7.283 - abs_cos_theta)
        else:
            h_nat = 1.810 * (abs_delta_T ** (1/3)) / (1.382 + abs_cos_theta)
        
    return h_nat

def conv_heat_transfer_coeff(T_surface, windspeed, T_ambient=None, tilt=0.0, model='Maestre-Valero', char_length=None):
    """
    Calculates convective heat transfer coefficient.
    T_surface: Surface temperature in Celsius
    windspeed: Wind speed in m/s
    T_ambient: Ambient air temperature in Celsius (optional, required for Combined-TARP models)
    tilt: Panel tilt angle in degrees (optional, required for Combined-TARP models)
    char_length: Characteristic length L = As/P in meters (optional, triggers size-dependent scaling)
    """
    if model == 'Maestre-Valero':
        T_ref = T_ambient if T_ambient is not None else T_surface
        return 7.6 + 6.6 * windspeed * (511 + 294) / (511 + T_ref + 273.15)
    elif model == 'Richards':
        T_ref = T_ambient if T_ambient is not None else T_surface
        return 5.9 + 4.1 * windspeed * (511 + 294) / (511 + T_ref + 273.15)
    elif model == 'Sharples':
        return 8 + 2.5 * windspeed
        
    # TARP/Combined models (combination of natural + forced component)
    if T_ambient is None:
        h_nat = 0.0
    else:
        h_nat = get_tarp_natural_convection(T_surface, T_ambient, tilt, char_length=char_length)
        
    # Scale forced components by (L_ref / L)^0.5
    # The reference length L_ref corresponds to the characteristic length of the panels in the base study.
    # Maestre-Valero: 1.0 m square foils -> L_ref = As/P = 0.25 m
    # Richards / Sharples: building roofs / large walls -> L_ref = 1.0 m
    if model == 'Combined-TARP-Maestre':
        L_ref = 0.25
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 6.6 * windspeed * (511 + 294) / (511 + T_surface + 273.15)
        return h_nat + forced_term * forced_scale
    elif model == 'Combined-TARP-Richards':
        L_ref = 1.0
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 4.1 * windspeed * (511 + 294) / (511 + T_surface + 273.15)
        return h_nat + forced_term * forced_scale
    elif model == 'Combined-TARP-Sharples':
        # Sharples (1998) used a 1.8m x 0.9m flat plate solar collector.
        # For a 1.8m x 0.9m panel: L_ref = As/P = (1.8 * 0.9) / (2 * (1.8 + 0.9)) = 0.3 m.
        L_ref = 0.3
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 2.5 * windspeed
        return h_nat + forced_term * forced_scale
    else:
        raise ValueError(f'Unknown model: {model}')

def Pconvection(T_amb, T_surf, windspeed, tilt=0.0, model='Maestre-Valero', char_length=None):
    hc = conv_heat_transfer_coeff(T_surface=T_surf, windspeed=windspeed, T_ambient=T_amb, tilt=tilt, model=model, char_length=char_length)
    return hc * (T_amb - T_surf)

def deriv_m(T_amb, H, T_surf, windspeed, p, tilt=0.0, model='Maestre-Valero', char_length=None):
    Ca = 1005

    Tdew = Tdewpoint(T_amb, H)
    if T_surf <= Tdew:
        hc = conv_heat_transfer_coeff(T_surface=T_surf, windspeed=windspeed, T_ambient=T_amb, tilt=tilt, model=model, char_length=char_length)
        val = ((hc * 0.622) / (Ca * p)) * (VPsat(Tdew) - VPsat(T_surf))
    else:
        val = 0
    return val

def VPsat(T_amb):
    """
    T_amb (float): Temperature in degrees celsius
    returns (float): vapor pressure in kPa
    """
    val = 0.61121 * np.exp((18.678 - T_amb / 234.5) * (T_amb / (257.14 + T_amb)))
    #val = saturation_vapor_pressure(T_amb*units.degC).to('kPa').magnitude # Super slow...

    return val

def Tdewpoint(T_amb, humidity):
    """
    T_amb (float): Temperature in degrees celsius
    humidity (float): Humidity in percent
    """
    return np.power(humidity/100,1/8)*(112+0.9*T_amb)+0.1*T_amb-112

def planck_law(wavelength, T_kelvin):
    """
    Wavelength in meters
    Temperature in Kelvin
    Calculates the spectral radiant exitance of a blackbody.
    Returns power per unit area per unit wavelength (W/m^3).
    """
    numerator = 3.746914726083474e-16 # 2 * np.pi * H_CONST * (C_CONST ** 2)
    HxCoKb = 0.014404347826086955 # H_CONST * C_CONST / Kb_CONST

    exponent = HxCoKb / ( wavelength * T_kelvin)
    exponent = np.clip(exponent, 0.1, 450) # Works for both scalars and arrays
    denominator = (wavelength ** 5) * (np.exp(exponent) - 1.0)

    return numerator / denominator

class emissivity_model(object):
    def __init__(self, LWIRemm=0.95, SWIRemm=0.02, model=None):
        self.LWIRemm = LWIRemm
        self.SWIRemm = SWIRemm

        if model == None:
            self.model = 'Tristram'

    def tristram_emissivity (self, wavelength):
        """
        wavelength (float or array-like): Wavelength in meters

        return (float or array-like): Emisivity of same shape as wavelength
        """
        wav = np.atleast_1d(wavelength)
        emis = np.full_like(wav, np.nan, dtype=float)

        visIR_mask = np.logical_and(wav >= 0.1e-6, wav <= 2.5e-6)
        midIR_mask = np.logical_and(wav > 2.5e-6, wav <= 8e-6)
        # Extended mask to cover all wavelengths > 8 µm to avoid NaN at far-IR wavelengths
        LWIR_mask = wav > 8e-6

        emis[visIR_mask] = self.SWIRemm
        emis[midIR_mask] = self.LWIRemm * np.sqrt((wav[midIR_mask] - 2.5e-6) / (8e-6 - 2.5e-6))
        emis[LWIR_mask] = self.LWIRemm

        if emis.size == 1:
            return emis[0]
        else:
            return emis

    def __call__(self, wavelength):
        if self.model == 'Tristram':
            return self.tristram_emissivity(wavelength)
        else:
            raise ValueError('Unknown model')

def spectral_humidCorr (wavelength, humidity):
    """
    :param wavelength (float or array): wavelength in m:
    :param humidity: Relative humidity in %:
    :return:

    Emperical correction factor used by Tristram.
    """
    wav = np.atleast_1d(wavelength)
    h_corr = np.full_like(wav, np.nan, dtype=float)

    visIR_mask = np.logical_and(wav >= 0.1e-6, wav <= 2.5e-6)
    midIR_mask = np.logical_and(wav > 2.5e-6, wav <= 8e-6)
    LWIR_mask = np.logical_and(wav > 8e-6, wav <= 13e-6)
    # Wavelengths > 13 µm act as a blackbody in the atmosphere due to H2O/CO2 absorption
    farIR_mask = wav > 13e-6

    h_corr[visIR_mask] = 0
    h_corr[midIR_mask] = 0.005*humidity
    h_corr[LWIR_mask] = 0.005*humidity
    h_corr[farIR_mask] = 1.0

    if h_corr.size == 1:
        return h_corr[0]
    else:
        return h_corr

class ThermalHeatTransfer (object):
    def __init__ (self, low_wavelength, high_wavelength, num_wavelengths,
                  surface=None,
                  humidity_model=None,
                  cloud_temperature_delTamb=0): #Assume clouds are at ambient temperature - should probably be -20IMPORTANT
        self.low_wavelength = low_wavelength
        self.high_wavelength = high_wavelength
        self.num_wavelengths = num_wavelengths
        self.cloud_temperature_delTamb = cloud_temperature_delTamb

        self._surface = surface

        if humidity_model is None:
            self.spectral_humidity_model = np.zeros_like
        else:
            self.spectral_humidity_model = humidity_model

        self.wavelengths = np.linspace(self.low_wavelength, self.high_wavelength, num_wavelengths)

        if self._surface is not None:
            self.absorptivities = self.surface.spectral_absorptivity(self.wavelengths)
            self.sky_view_factor = self._surface.sky_view_factor
        else:
            self.absorptivities = None
            self.sky_view_factor = 1

    @property
    def surface (self):
        return self._surface

    @surface.setter
    def surface(self, surface):
        self._surface = surface
        self.absorptivities = self._surface.spectral_absorptivity(self.wavelengths)
        self.sky_view_factor = self._surface.sky_view_factor
        print('Set spectral_absorptivity & sky_view_factor from surface')

    def update_surface_cache(self):
        """
        Manually syncs the thermal model with the surface.
        Call this immediately after modifying parameters of the assigned surface.
        """
        if self._surface is not None:
            self.absorptivities = self._surface.spectral_absorptivity(self.wavelengths)
            self.sky_view_factor = self._surface.sky_view_factor
        else:
            self.absorptivities = None
            self.sky_view_factor = 1

    def cloud_temperature(self, tamb):
        """
        tamb (float): Temperature in degrees celsius
        returns cloud temperature in degrees celsius
        """
        return tamb + self.cloud_temperature_delTamb

    def P_outgoing(self, T_surf):
        spectrum = planck_law(self.wavelengths, T_surf + 273.15) * self.absorptivities
        return trapezoid(y=spectrum, x=self.wavelengths)

    def P_incoming(self, T_amb, humidity=0, cloud_cover=0):
        assert self.sky_view_factor is not None, "Sky view factor must be set"
        spectrum1 = planck_law(self.wavelengths, T_amb + 273.15) * self.absorptivities
        integral_1 = trapezoid(y=spectrum1, x=self.wavelengths)

        humidity_spectrum = spectrum1 * self.spectral_humidity_model(self.wavelengths, humidity)
        if cloud_cover > 0.05:
            cloud_spectrum = planck_law(self.wavelengths, self.cloud_temperature(T_amb) + 273.15)
            spectrum2 = cloud_spectrum*cloud_cover + (1-cloud_cover)*humidity_spectrum
        else:
            spectrum2 = humidity_spectrum

         #Clearsky, incident radiation is due to humidity
        integral_2 = trapezoid(y=spectrum2, x=self.wavelengths)

        return (1-self.sky_view_factor) * integral_1 + (self.sky_view_factor) * integral_2

    def __call__(self, T_amb, T_surf, humidity=0, cloud_cover=0):
        return self.P_incoming(T_amb, humidity=humidity, cloud_cover=cloud_cover) - self.P_outgoing(T_surf)



def heat_balance_equation(Tsurf_guess, Tamb, irradiance, reflectivity, radht, humidity, windspeed, cloud_cover, cond_rate=0, model='Maestre-Valero', char_length=None):
    """
    The objective function that we want to equal zero.
    Calculates the net heat flux for a given guess of surface temperature (Tc_guess).
    """
    Lv = 2256*1e3 # latent heat of vapourisation

    # Calculate individual heat fluxes
    P_rad = radht(T_surf=Tsurf_guess[0], T_amb=Tamb, humidity=humidity, cloud_cover=cloud_cover)
    P_sol = (1 - reflectivity) * irradiance
    tilt = radht.surface.tilt if (hasattr(radht, 'surface') and radht.surface is not None) else 0.0
    P_conv = Pconvection(Tamb, Tsurf_guess[0], windspeed, tilt=tilt, model=model, char_length=char_length)
    P_lat = Lv * cond_rate # 0, unless condensation rate provided

    # print (f'LWIR out: {P_out:.1f}, LWIR in: {P_inc:.1f}, solar: {P_sol:.1f}')
    # Calculate net flux (Outgoing minus all incoming/generated heat)
    # At equilibrium, this should equal 0.
    net_flux = P_rad + P_sol + P_conv + P_lat

    return net_flux

def plot_simulation(df, relT=False):
    fig, [ax1, ax2] = plt.subplots(2,1, sharex=True)
    ax1t = ax1.twinx()
    ax1.set_zorder(ax1t.get_zorder() + 1)
    ax1.patch.set_visible(False)

    if relT == True:
        offset = df['Tamb']
        ybounds = [-7,7]
    else:
        offset = 0
        ybounds = [6,24]

    ax1t.fill_between(df.index, df['GHI'], color='xkcd:sun yellow', alpha=0.5, edgecolor=None)
    ax1.plot(df.index, df['Tamb']-offset, color='k', lw=1, ls=':')
    ax1.plot(df.index, df['Tsurf']-offset, color='k')
    ax1.plot(df.index, df['Theory - Tsurf']-offset, color='k', ls='--')

    # Plot experimental cumulative dew collected in L/m2
    ax2.plot(df.index, np.cumsum(df['Dew collected, mL/m2'])/1000, color='k', label='Experiment')
    # Plot theoretical cumulative condensation in L/m2 (which matches the unit of experimental cumsum)
    if 'Theory - cumulative condensation' in df.columns:
        ax2.plot(df.index, df['Theory - cumulative condensation'], color='k', ls='--', label='Theory')
    else:
        # Fallback if the user only integrated cond rate
        ax2.plot(df.index, df['Theory - cond rate'], color='k', ls='--', label='Theory (Rate)')
    ax2.legend()

    ax1.set(xlim=[df.index[1], df.index[-2]], ylim=ybounds, ylabel='Temperature, C')
    ax1t.set(ylim=[0,999], ylabel='Solar radiation [W/m2]')
    ax2.set(ylabel='Dew collected, L/m2')