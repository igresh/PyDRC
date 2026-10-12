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
                 char_length=None,
                 back_conductance=0.0,
                 insulation_conductivity=None,
                 insulation_thickness=None,
                 ground_h_eff=None,
                 ground_emissivity=0.95):
        """
        ground_h_eff (float or None): effective heat transfer coefficient of the surrounding ground,
            W/(m^2 K). Enables the GHI-driven ground temperature model (typ. 25-35 for concrete/paving).
            If None, the surroundings are assumed to be at ambient air temperature.
        ground_emissivity (float): broadband LWIR emissivity of the ground (typ. 0.90-0.97).
            The ground's solar albedo is taken from surrounds_albedo.
        """

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
        self._back_conductance = back_conductance
        self.insulation_conductivity = insulation_conductivity
        self.insulation_thickness = insulation_thickness
        self.ground_h_eff = ground_h_eff
        self.ground_emissivity = ground_emissivity

    @property
    def back_conductance(self):
        if self.insulation_conductivity is not None and self.insulation_thickness is not None and self.insulation_thickness > 0:
            return self.insulation_conductivity / self.insulation_thickness
        return self._back_conductance

    @back_conductance.setter
    def back_conductance(self, value):
        self._back_conductance = value

    @property
    def sky_view_factor(self):
        if self._sky_view_factor is not None:
            return self._sky_view_factor
        else:
            # Isotropic view factor from a tilted plane to the sky hemisphere,
            # assuming an unobstructed horizon over flat ground: F_sky = (1 + cos β) / 2
            return (1 + np.cos(np.deg2rad(self.tilt))) / 2

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
        # GHI is used for the ground-temperature model; optional for backwards compatibility
        self.interp_GHI = self.create_interpolator('GHI') if 'GHI' in self.exp_df.columns else None

        self.heatcap_water = 4184

    def create_interpolator(self, name):
        return interp1d(self.exp_df['Time, s'], self.exp_df[name], kind='linear', bounds_error=False, fill_value=(self.exp_df[name].iloc[0], self.exp_df[name].iloc[-1]))

    def _ghi(self, t):
        return float(self.interp_GHI(t)) if self.interp_GHI is not None else None

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
                                                           cond_rate=cond_rate, model=self.surface.convection_model, char_length=self.surface.char_length,
                                                           back_conductance=getattr(self.surface, 'back_conductance', 0.0),
                                                           ghi=self._ghi(t))
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

        # Modelled ground temperature (equals Tamb if ground_h_eff is None or GHI unavailable)
        sim_df['Theory - Tground'] = [
            self.Prad_function.ground_temperature(float(self.interp_Tamb(t)), ghi=self._ghi(t),
                                                  humidity=float(self.interp_Humid(t)),
                                                  cloud_cover=float(self.interp_cloud(t)))
            for t in t_exp
        ]

        # Modelled downwelling atmospheric thermal radiation (W/m^2) on a horizontal blackbody
        # In-band: integrated over the panel's simulation wavelength range (e.g. 2.5 - 25 um)
        sim_df['Theory - downwelling_sky_flux'] = [
            self.Prad_function.downwelling_sky_radiation(
                float(self.interp_Tamb(t)),
                humidity=float(self.interp_Humid(t)),
                cloud_cover=float(self.interp_cloud(t)),
                broadband=False
            )
            for t in t_exp
        ]
        # Broadband: integrated over the full meteorological thermal IR spectrum (2.5 - 100 um, directly comparable to ERA5 strd)
        sim_df['Theory - strd_broadband'] = [
            self.Prad_function.downwelling_sky_radiation(
                float(self.interp_Tamb(t)),
                humidity=float(self.interp_Humid(t)),
                cloud_cover=float(self.interp_cloud(t)),
                broadband=True
            )
            for t in t_exp
        ]
        sim_df['Theory - strd'] = sim_df['Theory - strd_broadband']

        # If measured or ERA5 strd is present in the dataset, compute diagnostic residuals
        if 'strd' in self.exp_df.columns:
            sim_df['strd'] = self.exp_df['strd'].values
            sim_df['Diagnostic - strd_error_broadband'] = sim_df['Theory - strd_broadband'] - sim_df['strd']
            sim_df['Diagnostic - strd_error_inband'] = sim_df['Theory - downwelling_sky_flux'] - sim_df['strd']

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

        ghi_basis = np.linspace(0,1300,13)
        Tsurfs = []
        initial_guess = [self.Tamb]

        # Step 2: Use fsolve to find the exact Tc that makes the heat_balance_equation return 0.
        # We pass all the environmental variables through the 'args' tuple.
        for i_ghi in ghi_basis:
            Tc_equilibrium, = fsolve(
                heat_balance_equation,
                x0=initial_guess,
                args=(self.Tamb, i_ghi, self.surface.solar_reflectivity, self.radht, self.humidity, self.windspeed, self.cloud_cover, 0.0, self.surface.convection_model, self.surface.char_length, getattr(self.surface, 'back_conductance', 0.0), i_ghi)
            )
            Tsurfs.append(Tc_equilibrium)
        GHI_interp = interp1d(ghi_basis, np.array(Tsurfs), kind='linear')

        return GHI_interp(ghi) - self.Tamb

    def T_ghi_LWIRemm(self, ghi, reflectivity, LWIRemm):
        self.surface.solar_reflectivity = reflectivity
        self.surface.spectral_absorptivity.LWIRemm = LWIRemm
        self.surface.spectral_absorptivity.SWIRemm = 1-reflectivity
        self.radht.update_surface_cache()

        ghi_basis = np.linspace(0,1300,13)
        Tsurfs = []
        initial_guess = [self.Tamb]

        # Step 2: Use fsolve to find the exact Tc that makes the heat_balance_equation return 0.
        # We pass all the environmental variables through the 'args' tuple.
        for i_ghi in ghi_basis:
            Tc_equilibrium, = fsolve(
                heat_balance_equation,
                x0=initial_guess,
                args=(self.Tamb, i_ghi, self.surface.solar_reflectivity, self.radht, self.humidity, self.windspeed, self.cloud_cover, 0.0, self.surface.convection_model, self.surface.char_length, getattr(self.surface, 'back_conductance', 0.0), i_ghi)
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
    new_df['Tsurf'] = df_section[(surface.name, 'Temp avg delta')] + df_section[('General', 'Air temp, ˚C')]
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

    strd_col = ('ERA5_Thermal', 'Surface long-wave (thermal) radiation downwards, W/m^2')


    if strd_col is not None:
        raw_strd = df_section[strd_col].copy()
        # ERA5 hourly accumulated data is reported in J/m^2 (typical magnitude ~ 1e6).
        # Divide by 3600 seconds to convert to mean flux in W/m^2.
        col_name_str = str(strd_col)
        if 'J/m^2' in col_name_str or (pd.api.types.is_numeric_dtype(raw_strd) and raw_strd.dropna().mean() > 5000):
            new_df['strd'] = raw_strd / 3600.0
        else:
            new_df['strd'] = raw_strd


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

def Pconduction(T_amb, T_surf, back_conductance=0.0):
    """
    Calculates conductive heat transfer flux with ambient / boundary in W/m^2.
    back_conductance is in W/(m^2 K).
    Positive flux corresponds to heat entering the surface from ambient.
    """
    return back_conductance * (T_amb - T_surf)

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

    def sky_spectrum(self, T_amb, humidity=0, cloud_cover=0):
        """
        Downwelling spectral sky radiance (W/m^3) incident on a horizontal blackbody,
        i.e. NOT weighted by the panel's spectral absorptivity.
        """
        bb_amb = planck_law(self.wavelengths, T_amb + 273.15)
        humidity_spectrum = bb_amb * self.spectral_humidity_model(self.wavelengths, humidity)
        if cloud_cover > 0.05:
            cloud_spectrum = planck_law(self.wavelengths, self.cloud_temperature(T_amb) + 273.15)
            return cloud_spectrum*cloud_cover + (1-cloud_cover)*humidity_spectrum
        return humidity_spectrum

    def ground_temperature(self, T_amb, ghi=None, humidity=0, cloud_cover=0):
        """
        Quasi-steady, linearised energy balance of horizontal ground (Celsius):

            T_g = T_amb + [(1 - albedo) * GHI - eps_g * (L_bb(T_amb) - L_sky)] / h_eff

        albedo  : surface.surrounds_albedo (same value used for ground-reflected solar)
        eps_g   : surface.ground_emissivity
        h_eff   : surface.ground_h_eff, W/(m^2 K); lumps convection, linearised radiation
                  and conduction into the ground. If None, T_g = T_amb (legacy behaviour).
        L_bb, L_sky are integrated over the same wavelength range as the panel model.
        """
        surf = self._surface
        h_eff = getattr(surf, 'ground_h_eff', None) if surf is not None else None
        if ghi is None or h_eff is None or h_eff <= 0:
            return T_amb

        albedo = getattr(surf, 'surrounds_albedo', 0.15)
        eps_g = getattr(surf, 'ground_emissivity', 0.95)

        L_bb_amb = trapezoid(y=planck_law(self.wavelengths, T_amb + 273.15), x=self.wavelengths)
        L_sky = trapezoid(y=self.sky_spectrum(T_amb, humidity, cloud_cover), x=self.wavelengths)

        net_lw_loss = eps_g * (L_bb_amb - L_sky)
        return T_amb + ((1 - albedo) * max(ghi, 0.0) - net_lw_loss) / h_eff

    def downwelling_sky_radiation(self, T_amb, humidity=0, cloud_cover=0, broadband=True):
        """
        Calculates downwelling longwave radiation from the sky on a horizontal blackbody (W/m^2).
        
        Parameters:
        -----------
        T_amb : float
            Ambient temperature in Celsius.
        humidity : float
            Relative humidity (%).
        cloud_cover : float
            Cloud cover fraction (0.0 to 1.0).
        broadband : bool, default True
            If True, integrates across the full meteorological thermal infrared spectrum (2.5 - 100 um),
            directly corresponding to ERA5 'strd' or a broadband pyrgeometer.
            If False, integrates across self.wavelengths (the panel's simulation band).
        """
        if not broadband:
            spectrum = self.sky_spectrum(T_amb, humidity=humidity, cloud_cover=cloud_cover)
            return trapezoid(y=spectrum, x=self.wavelengths)
        else:
            if not hasattr(self, '_broadband_wavs'):
                self._broadband_wavs = np.linspace(2.5e-6, 100.0e-6, 300)
            wavs = self._broadband_wavs
            bb_amb = planck_law(wavs, T_amb + 273.15)
            h_spec = bb_amb * self.spectral_humidity_model(wavs, humidity)
            if cloud_cover > 0.05:
                cloud_spec = planck_law(wavs, self.cloud_temperature(T_amb) + 273.15)
                full_sky = cloud_spec * cloud_cover + (1.0 - cloud_cover) * h_spec
            else:
                full_sky = h_spec
            return trapezoid(y=full_sky, x=wavs)

    def effective_sky_emissivity(self, T_amb, humidity=0, cloud_cover=0, broadband=True):
        """
        Calculates effective broadband sky emissivity: eps_sky = L_sky / (sigma * T_amb^4).
        """
        sigma = 5.670374e-8
        T_k = T_amb + 273.15
        L_sky = self.downwelling_sky_radiation(T_amb, humidity=humidity, cloud_cover=cloud_cover, broadband=broadband)
        return L_sky / (sigma * (T_k ** 4))

    def P_incoming(self, T_amb, humidity=0, cloud_cover=0, ghi=None):
        assert self.sky_view_factor is not None, "Sky view factor must be set"
        # Surroundings (ground) term: blackbody at the modelled ground temperature
        T_ground = self.ground_temperature(T_amb, ghi=ghi, humidity=humidity, cloud_cover=cloud_cover)
        spectrum1 = planck_law(self.wavelengths, T_ground + 273.15) * self.absorptivities
        integral_1 = trapezoid(y=spectrum1, x=self.wavelengths)

        # Sky term. Clearsky, incident radiation is due to humidity
        spectrum2 = self.sky_spectrum(T_amb, humidity, cloud_cover) * self.absorptivities
        integral_2 = trapezoid(y=spectrum2, x=self.wavelengths)

        return (1-self.sky_view_factor) * integral_1 + (self.sky_view_factor) * integral_2

    def __call__(self, T_amb, T_surf, humidity=0, cloud_cover=0, ghi=None):
        return self.P_incoming(T_amb, humidity=humidity, cloud_cover=cloud_cover, ghi=ghi) - self.P_outgoing(T_surf)



def heat_balance_equation(Tsurf_guess, Tamb, irradiance, reflectivity, radht, humidity, windspeed, cloud_cover, cond_rate=0, model='Maestre-Valero', char_length=None, back_conductance=None, ghi=None):
    """
    The objective function that we want to equal zero.
    Calculates the net heat flux for a given guess of surface temperature (Tc_guess).
    ghi: global horizontal irradiance (W/m^2), used to estimate the ground temperature.
         If None, the ground is assumed to be at ambient temperature.
    """
    Lv = 2256*1e3 # latent heat of vapourisation

    # Calculate individual heat fluxes
    P_rad = radht(T_surf=Tsurf_guess[0], T_amb=Tamb, humidity=humidity, cloud_cover=cloud_cover, ghi=ghi)
    P_sol = (1 - reflectivity) * irradiance
    tilt = radht.surface.tilt if (hasattr(radht, 'surface') and radht.surface is not None) else 0.0
    P_conv = Pconvection(Tamb, Tsurf_guess[0], windspeed, tilt=tilt, model=model, char_length=char_length)
    P_lat = Lv * cond_rate # 0, unless condensation rate provided

    # Conductive heat flux through insulation with ambient air (T_boundary = T_amb)
    if back_conductance is None:
        if hasattr(radht, 'surface') and radht.surface is not None:
            back_conductance = getattr(radht.surface, 'back_conductance', 0.0)
        else:
            back_conductance = 0.0
    P_cond = Pconduction(Tamb, Tsurf_guess[0], back_conductance=back_conductance)

    # print (f'LWIR out: {P_out:.1f}, LWIR in: {P_inc:.1f}, solar: {P_sol:.1f}')
    # Calculate net flux (Outgoing minus all incoming/generated heat)
    # At equilibrium, this should equal 0.
    net_flux = P_rad + P_sol + P_conv + P_lat + P_cond

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


def diagnose_strd(df, radht=None, plot=True, title='Downwelling Longwave Radiation (strd) Diagnostic'):
    """
    Diagnostic tool comparing modelled downwelling atmospheric thermal radiation against measured/ERA5 'strd'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from make_dataframe or run_simulation.
        Must contain 'Tamb', 'Humidity', 'Cloud cover'.
        If 'strd' is present, full statistical comparisons are generated.
    radht : ThermalHeatTransfer, optional
        Used to compute 'Theory - strd_broadband' and 'Theory - downwelling_sky_flux'
        if they are not already present in df.
    plot : bool, default True
        Whether to generate and return diagnostic plots.
    title : str, default 'Downwelling Longwave Radiation (strd) Diagnostic'
        Title for the diagnostic plot.
        
    Returns:
    --------
    stats : dict
        Statistical error metrics and performance breakdown.
    fig : matplotlib.figure.Figure (optional, only if plot=True)
    """
    df = df.copy()

    # If theory columns are missing, compute them using radht if available
    if 'Theory - strd_broadband' not in df.columns or 'Theory - downwelling_sky_flux' not in df.columns:
        if radht is None:
            raise ValueError(
                "Neither 'Theory - strd_broadband' is in df nor was a 'radht' (ThermalHeatTransfer) instance provided. "
                "Please run run_simulation() first or pass radht=..."
            )
        df['Theory - downwelling_sky_flux'] = [
            radht.downwelling_sky_radiation(row['Tamb'], humidity=row['Humidity'], cloud_cover=row['Cloud cover'], broadband=False)
            for _, row in df.iterrows()
        ]
        df['Theory - strd_broadband'] = [
            radht.downwelling_sky_radiation(row['Tamb'], humidity=row['Humidity'], cloud_cover=row['Cloud cover'], broadband=True)
            for _, row in df.iterrows()
        ]

    has_measured = 'strd' in df.columns and not df['strd'].dropna().empty

    stats = {
        'n_samples': len(df),
        'mean_theory_broadband': float(df['Theory - strd_broadband'].mean()),
        'mean_theory_inband': float(df['Theory - downwelling_sky_flux'].mean()),
        'has_measured_strd': has_measured
    }

    if has_measured:
        valid_mask = df['strd'].notnull() & df['Theory - strd_broadband'].notnull()
        valid_df = df[valid_mask]
        
        diff_broadband = valid_df['Theory - strd_broadband'] - valid_df['strd']
        diff_inband = valid_df['Theory - downwelling_sky_flux'] - valid_df['strd']

        stats['mean_measured_strd'] = float(valid_df['strd'].mean())
        stats['mbe_broadband'] = float(diff_broadband.mean())
        stats['rmse_broadband'] = float(np.sqrt((diff_broadband ** 2).mean()))
        stats['mbe_inband'] = float(diff_inband.mean())
        stats['rmse_inband'] = float(np.sqrt((diff_inband ** 2).mean()))

        # Clear-sky subset (cloud <= 0.1) -> Isolates atmospheric water vapor & humidity model
        clear_mask = valid_df['Cloud cover'] <= 0.1
        if clear_mask.any():
            stats['n_clear_sky'] = int(clear_mask.sum())
            stats['clear_sky_mean_measured'] = float(valid_df.loc[clear_mask, 'strd'].mean())
            stats['clear_sky_mean_model'] = float(valid_df.loc[clear_mask, 'Theory - strd_broadband'].mean())
            stats['clear_sky_mbe'] = float(diff_broadband[clear_mask].mean())
            stats['clear_sky_rmse'] = float(np.sqrt((diff_broadband[clear_mask] ** 2).mean()))

        # Overcast subset (cloud >= 0.8) -> Isolates cloud temperature & emissivity model
        overcast_mask = valid_df['Cloud cover'] >= 0.8
        if overcast_mask.any():
            stats['n_overcast'] = int(overcast_mask.sum())
            stats['overcast_mean_measured'] = float(valid_df.loc[overcast_mask, 'strd'].mean())
            stats['overcast_mean_model'] = float(valid_df.loc[overcast_mask, 'Theory - strd_broadband'].mean())
            stats['overcast_mbe'] = float(diff_broadband[overcast_mask].mean())
            stats['overcast_rmse'] = float(np.sqrt((diff_broadband[overcast_mask] ** 2).mean()))

    # Print summary report
    print("=" * 65)
    print("         DOWNWELLING LONGWAVE (strd) DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"Total samples: {stats['n_samples']}")
    print(f"Modelled downwelling (broadband, 2.5-100 um): {stats['mean_theory_broadband']:.1f} W/m^2")
    print(f"Modelled downwelling (in-band):              {stats['mean_theory_inband']:.1f} W/m^2")

    if has_measured:
        print(f"Measured/ERA5 strd downwelling:             {stats['mean_measured_strd']:.1f} W/m^2")
        print(f"Overall Broadband Bias (MBE):               {stats['mbe_broadband']:+.1f} W/m^2")
        print(f"Overall Broadband RMSE:                     {stats['rmse_broadband']:.1f} W/m^2")
        print("-" * 65)
        print("PHYSICAL BREAKDOWN:")
        if 'n_clear_sky' in stats:
            print(f"* Clear-sky (Cloud <= 0.1, N={stats['n_clear_sky']}):")
            print(f"  - ERA5: {stats['clear_sky_mean_measured']:.1f} W/m^2 | Model: {stats['clear_sky_mean_model']:.1f} W/m^2")
            print(f"  - MBE: {stats['clear_sky_mbe']:+.1f} W/m^2 | RMSE: {stats['clear_sky_rmse']:.1f} W/m^2")
            if stats['clear_sky_mbe'] < -20:
                print("  -> Underestimating clear sky longwave (air is more opaque than modeled).")
            elif stats['clear_sky_mbe'] > 20:
                print("  -> Overestimating clear sky longwave (air is more transparent than modeled).")
        if 'n_overcast' in stats:
            print(f"* Overcast (Cloud >= 0.8, N={stats['n_overcast']}):")
            print(f"  - ERA5: {stats['overcast_mean_measured']:.1f} W/m^2 | Model: {stats['overcast_mean_model']:.1f} W/m^2")
            print(f"  - MBE: {stats['overcast_mbe']:+.1f} W/m^2 | RMSE: {stats['overcast_rmse']:.1f} W/m^2")
    else:
        print("\nNote: 'strd' column not detected in DataFrame.")
        print("Once you download ERA5 'strd' and load it into your DataFrame, rerun")
        print("diagnose_strd(df) to see full bias, RMSE, clear-sky vs overcast breakdown.")
    print("=" * 65)

    if not plot:
        return stats

    # Diagnostic Plots
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_time = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_resid = fig.add_subplot(gs[1, 1])

    # 1. Time series plot
    if has_measured:
        ax_time.plot(df.index, df['strd'], label='ERA5 / Measured strd', color='tab:blue', lw=2)
    ax_time.plot(df.index, df['Theory - strd_broadband'], label='Model (Broadband)', color='tab:orange', lw=1.8, ls='--')
    ax_time.plot(df.index, df['Theory - downwelling_sky_flux'], label='Model (In-band)', color='tab:green', lw=1.2, ls=':')
    ax_time.set_ylabel('Downwelling Flux [W/m²]')
    ax_time.set_title(title, fontweight='bold', fontsize=12)

    ax_cloud = ax_time.twinx()
    ax_cloud.fill_between(df.index, df['Cloud cover'], alpha=0.15, color='gray', label='Cloud cover')
    ax_cloud.set_ylabel('Cloud Cover [0-1]', color='gray')
    ax_cloud.set_ylim([0, 1.2])

    lines_1, labels_1 = ax_time.get_legend_handles_labels()
    lines_2, labels_2 = ax_cloud.get_legend_handles_labels()
    ax_time.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.85)

    if has_measured:
        # 2. Scatter plot (Model vs Measured)
        sc = ax_scatter.scatter(valid_df['strd'], valid_df['Theory - strd_broadband'],
                                c=valid_df['Humidity'], cmap='viridis', alpha=0.75, edgecolors='none', s=25)
        cbar = plt.colorbar(sc, ax=ax_scatter)
        cbar.set_label('RH [%]')

        min_val = min(valid_df['strd'].min(), valid_df['Theory - strd_broadband'].min()) - 15
        max_val = max(valid_df['strd'].max(), valid_df['Theory - strd_broadband'].max()) + 15
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.2, label='1:1 Line')
        ax_scatter.set_xlim([min_val, max_val])
        ax_scatter.set_ylim([min_val, max_val])
        ax_scatter.set_xlabel('ERA5 / Measured strd [W/m²]')
        ax_scatter.set_ylabel('Modelled Broadband strd [W/m²]')
        ax_scatter.set_title(f'1:1 Comparison (RMSE={stats["rmse_broadband"]:.1f} W/m²)')
        ax_scatter.legend(loc='upper left')

        # 3. Residuals vs Relative Humidity
        diff = valid_df['Theory - strd_broadband'] - valid_df['strd']
        ax_resid.scatter(valid_df['Humidity'], diff, c=valid_df['Cloud cover'], cmap='coolwarm', alpha=0.75, s=25)
        ax_resid.axhline(0, color='k', ls='--', lw=1.2)
        
        # Fit trendline against humidity if enough points
        if len(valid_df) > 5 and valid_df['Humidity'].nunique() > 2:
            p = np.polyfit(valid_df['Humidity'], diff, 1)
            rh_line = np.linspace(valid_df['Humidity'].min(), valid_df['Humidity'].max(), 50)
            ax_resid.plot(rh_line, np.polyval(p, rh_line), color='crimson', lw=1.5, label=f'Trend ({p[0]:+.2f} W/m² per %RH)')
            ax_resid.legend(loc='best')

        ax_resid.set_xlabel('Relative Humidity [%]')
        ax_resid.set_ylabel('Error: Model - Measured [W/m²]')
        ax_resid.set_title('Residuals vs Humidity (Color = Cloud Cover)')
    else:
        ax_scatter.text(0.5, 0.5, "No 'strd' data available to plot scatter.", ha='center', va='center')
        ax_resid.text(0.5, 0.5, "No 'strd' data available to plot residuals.", ha='center', va='center')

    plt.tight_layout()
    return stats, fig