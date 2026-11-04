"""
Building.py: a simple single-storey house with a ventilated attic and a radiatively-cooled (or not) roof.

Couples the PyDRC roof-surface physics in Model.py (spectral long-wave exchange with the sky,
solar absorption, convection) to lumped attic-air and indoor-air nodes, so the effect of roof
paint on indoor temperature can be shown "all other things being equal".

Thermal network (per house)
---------------------------

          sun   sky/ground LWIR   wind
            \        |           /
     [ roof outer surface, T_roof ]  -- C_roof (per m² roof)
                     |
               U_roof_attic (roofing + underside film, per m² roof)
                     |
     [ attic air, T_attic ]  -- C_attic          ---- attic ventilation (rho cp V_attic ACH) to T_amb
                     |
               U_ceiling (ceiling + insulation, per m² floor)
                     |
     [ indoor air + furnishings, T_in ]  -- C_in
          |                 |                |
     walls & windows    ventilation     internal + window solar gains
       (UA to T_amb)   (rho cp V ACH)

Roof surface (W per m² roof):
    C_roof dT_roof/dt = (1-R)·G + P_rad(T_roof) + h_c·(T_amb - T_roof) + U_ra·(T_attic - T_roof)
Attic air (W):
    C_attic dT_attic/dt = U_ra·A_roof·(T_roof - T_attic) + K_attic_vent·(T_amb - T_attic) + U_c·A_floor·(T_in - T_attic)
Indoor air (W):
    C_in dT_in/dt = U_c·A_floor·(T_attic - T_in) + K_amb·(T_amb - T_in) + Q_internal + Q_window

P_rad and h_c come straight from Model.ThermalHeatTransfer and Model.Pconvection.
The insulation sits on the ceiling (between attic and house), not under the roof sheeting.
Attic ventilation defaults to 5 air changes per hour (a mid-range Australian roof space).

Simplifications (deliberate, for a clear "same house, different roof" comparison):
  * Irradiance on the roof is taken as GHI (a pitched roof's two slopes roughly average to this).
  * Walls are treated as being at air temperature on the outside (no wall sol-air heating).
  * Radiation from the roof underside to the top of the ceiling is lumped into U_roof_attic
    (i.e. it is treated as heating the attic node), and there is no reflective foil / sarking.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import Model

RHO_CP_AIR = 1.2 * 1005.0  # volumetric heat capacity of air, J/(m³·K)

# Roof paints used in the demo: solar reflectivity R and thermal (LWIR) emissivity eps
PAINTS = {
    'black':      dict(name='Black',        solar_reflectivity=0.05, thermal_emissivity=0.92),
    'charcoal':   dict(name='Charcoal',     solar_reflectivity=0.15, thermal_emissivity=0.92),
    'terracotta': dict(name='Terracotta',   solar_reflectivity=0.30, thermal_emissivity=0.92),
    'light_grey': dict(name='Light grey',   solar_reflectivity=0.55, thermal_emissivity=0.92),
    'white':      dict(name='White paint',  solar_reflectivity=0.85, thermal_emissivity=0.92),
    'cool':       dict(name='Cool paint (PDRC)', solar_reflectivity=0.96, thermal_emissivity=0.95),
}

# Ceiling U-values, attic air -> indoor air (incl. both air films), W/(m²·K) per m² of floor.
INSULATION = {
    'none': 2.5,   # plasterboard ceiling only
    'some': 0.8,   # ceiling with thin batts (~R1)
    'lots': 0.28,  # ceiling with thick batts (~R3.5)
}


class House(object):
    """
    Geometry and construction of a simple single-storey house with a gable roof over a square
    floor plan. Everything except the roof paint lives here, so two houses built from the same
    House differ only in their roof.
    """
    def __init__(self,
                 floor_area=100.0,          # m²
                 roof_pitch=20.0,           # degrees from horizontal
                 wall_height=2.4,           # m
                 wall_U=1.0,                # W/(m²·K)
                 window_area=15.0,          # m²
                 window_U=5.8,              # W/(m²·K), single glazing
                 window_solar_factor=0.05,  # W of indoor gain per (W/m² GHI) per m² window (shaded windows)
                 air_changes_per_hour=1.0,  # house ventilation, 1/h
                 internal_gains=400.0,      # W (people, appliances)
                 insulation='some',         # key of INSULATION, or a ceiling U-value in W/(m²·K)
                 roof_attic_U=6.0,          # W/(m²·K) per m² roof: roofing + underside convection/radiation
                 attic_air_changes_per_hour=5.0,  # attic ventilation, 1/h. Typical Australian roof spaces:
                                                  # ~1-3 sarked metal / sealed eaves, ~3-5 vented metal or
                                                  # sarked tiles, ~10-20+ unsarked tiles or whirlybirds.
                 roof_heat_capacity=10e3,   # J/(m²·K) per m² roof (outer roofing layer)
                 attic_heat_capacity=5e3,   # J/(m²·K) per m² floor (attic air, trusses, top of ceiling)
                 interior_heat_capacity=60e3):  # J/(m²·K) per m² floor (air, furniture, linings)
        self.floor_area = floor_area
        self.roof_pitch = roof_pitch
        self.wall_height = wall_height
        self.wall_U = wall_U
        self.window_area = window_area
        self.window_U = window_U
        self.window_solar_factor = window_solar_factor
        self.air_changes_per_hour = air_changes_per_hour
        self.internal_gains = internal_gains
        self.insulation = insulation
        self.roof_attic_U = roof_attic_U
        self.attic_air_changes_per_hour = attic_air_changes_per_hour
        self.roof_heat_capacity = roof_heat_capacity
        self.attic_heat_capacity = attic_heat_capacity
        self.interior_heat_capacity = interior_heat_capacity

    # ---------- geometry ----------
    @property
    def side(self):
        return np.sqrt(self.floor_area)

    @property
    def roof_area(self):
        return self.floor_area / np.cos(np.deg2rad(self.roof_pitch))

    @property
    def volume(self):
        return self.floor_area * self.wall_height

    @property
    def attic_volume(self):
        """Gable attic over a square floor: triangular prism."""
        ridge_height = (self.side / 2) * np.tan(np.deg2rad(self.roof_pitch))
        return 0.5 * self.floor_area * ridge_height

    @property
    def wall_area(self):
        return 4 * self.side * self.wall_height - self.window_area

    @property
    def roof_char_length(self):
        """L = area / perimeter of one roof slope (used for size-aware convection)."""
        slope = (self.side / 2) / np.cos(np.deg2rad(self.roof_pitch))
        return (self.side * slope) / (2 * (self.side + slope))

    # ---------- conductances (W/K) ----------
    @property
    def ceiling_U(self):
        if isinstance(self.insulation, str):
            return INSULATION[self.insulation]
        return float(self.insulation)

    @property
    def K_roof_attic(self):
        return self.roof_attic_U * self.roof_area

    @property
    def K_attic_vent(self):
        return RHO_CP_AIR * self.attic_volume * self.attic_air_changes_per_hour / 3600.0

    @property
    def K_ceiling(self):
        return self.ceiling_U * self.floor_area

    @property
    def K_ambient(self):
        """Indoor air -> outdoor air via walls, windows and house ventilation."""
        ventilation = RHO_CP_AIR * self.volume * self.air_changes_per_hour / 3600.0
        return self.wall_U * self.wall_area + self.window_U * self.window_area + ventilation

    # ---------- heat capacities (J/K) ----------
    @property
    def C_attic(self):
        return self.attic_heat_capacity * self.floor_area

    @property
    def C_in(self):
        return self.interior_heat_capacity * self.floor_area

    def gains(self, ghi):
        """Heat gains to indoor air that do not depend on the roof, W."""
        return self.internal_gains + self.window_solar_factor * self.window_area * max(ghi, 0.0)


class RoofedHouse(object):
    """
    A House with a particular roof finish. Uses Model.Surface / Model.ThermalHeatTransfer for the
    roof's radiative exchange and Model.Pconvection for its convection.
    """
    def __init__(self, house, paint='cool', solar_reflectivity=None, thermal_emissivity=None,
                 num_wavelengths=200, convection_model='Combined-TARP-Maestre',
                 boundary_layer='turbulent', ground_h_eff=30.0, cloud_temperature_delTamb=-8.0):
        p = PAINTS[paint] if paint is not None else {}
        R = solar_reflectivity if solar_reflectivity is not None else p['solar_reflectivity']
        eps = thermal_emissivity if thermal_emissivity is not None else p['thermal_emissivity']
        self.house = house
        self.paint = paint
        self.surface = Model.Surface(name=p.get('name', 'roof'),
                                     solar_reflectivity=R, thermal_emissivity=eps,
                                     tilt=house.roof_pitch, convection_model=convection_model,
                                     char_length=house.roof_char_length, boundary_layer=boundary_layer,
                                     ground_h_eff=ground_h_eff, transposition_model='haydavies')
        self.radht = Model.ThermalHeatTransfer(low_wavelength=2.5e-6, high_wavelength=50e-6,
                                               num_wavelengths=num_wavelengths, surface=self.surface,
                                               humidity_model=Model.spectral_humidCorr,
                                               cloud_temperature_delTamb=cloud_temperature_delTamb)

    def set_paint(self, solar_reflectivity=None, thermal_emissivity=None):
        if solar_reflectivity is not None:
            self.surface.solar_reflectivity = solar_reflectivity
        if thermal_emissivity is not None:
            self.surface.thermal_emissivity = thermal_emissivity
        self.radht.update_surface_cache()

    # ---------- fluxes ----------
    def roof_fluxes(self, T_roof, T_attic, T_amb, ghi, humidity, windspeed, cloud_cover=0.0):
        """Individual heat flows INTO the roof surface, W per m² of roof."""
        s = self.surface
        solar = (1 - s.solar_reflectivity) * max(ghi, 0.0)
        radiative = self.radht(T_amb=T_amb, T_surf=T_roof, humidity=humidity,
                               cloud_cover=cloud_cover, ghi=ghi)
        convective = Model.Pconvection(T_amb, T_roof, windspeed, tilt=s.tilt, model=s.convection_model,
                                       char_length=s.char_length, boundary_layer=s.boundary_layer)
        from_attic = self.house.roof_attic_U * (T_attic - T_roof)
        return dict(solar=solar, radiative=radiative, convective=convective, from_attic=from_attic)

    def _attic_and_indoor_steady(self, T_roof, T_amb, ghi):
        """Given the roof temperature, the attic and indoor nodes are linear: solve the 2x2 system."""
        h = self.house
        Kra, Kav, Kc, Ka = h.K_roof_attic, h.K_attic_vent, h.K_ceiling, h.K_ambient
        A = np.array([[Kra + Kav + Kc, -Kc],
                      [-Kc, Kc + Ka]])
        b = np.array([Kra * T_roof + Kav * T_amb,
                      Ka * T_amb + h.gains(ghi)])
        T_attic, T_in = np.linalg.solve(A, b)
        return T_attic, T_in

    # ---------- steady state ----------
    def steady_state(self, T_amb, ghi, humidity=50.0, windspeed=2.0, cloud_cover=0.0):
        """
        Equilibrium roof, attic and indoor temperatures for constant weather.
        This is the 'if the sun stayed like this all day' answer: good for the roof, but it
        overstates the indoor temperature because it ignores the house's thermal mass.
        """
        def net(T_roof):
            T_attic, _ = self._attic_and_indoor_steady(T_roof, T_amb, ghi)
            return sum(self.roof_fluxes(T_roof, T_attic, T_amb, ghi, humidity, windspeed, cloud_cover).values())

        T_roof = brentq(net, T_amb - 60.0, T_amb + 120.0, xtol=1e-4)
        T_attic, T_in = self._attic_and_indoor_steady(T_roof, T_amb, ghi)
        fluxes = self.roof_fluxes(T_roof, T_attic, T_amb, ghi, humidity, windspeed, cloud_cover)
        h = self.house
        return dict(T_roof=T_roof, T_attic=T_attic, T_in=T_in, T_amb=T_amb,
                    ceiling_heat_into_house_W=h.K_ceiling * (T_attic - T_in),
                    attic_vent_heat_removed_W=h.K_attic_vent * (T_attic - T_amb),
                    **{f'flux_{k}': v for k, v in fluxes.items()})

    # ---------- transient ----------
    def simulate(self, weather, T0=None, spinup_days=2):
        """
        Transient simulation driven by a weather DataFrame indexed by hours (float) with columns
        'T_amb' (°C), 'ghi' (W/m²), 'humidity' (%), 'windspeed' (m/s) and optionally 'cloud_cover'.
        The weather is repeated for `spinup_days` beforehand so the result is a periodic steady state.
        Returns a DataFrame with T_roof, T_attic, T_in and the ceiling heat flow into the house.
        """
        t_h = np.asarray(weather.index, dtype=float)
        period = t_h[-1] - t_h[0]
        cols = {c: np.asarray(weather[c], dtype=float) for c in weather.columns}
        if 'cloud_cover' not in cols:
            cols['cloud_cover'] = np.zeros_like(t_h)

        def wx(t_hours):
            tt = t_h[0] + np.mod(t_hours - t_h[0], period) if period > 0 else t_h[0]
            return {c: float(np.interp(tt, t_h, v)) for c, v in cols.items()}

        h = self.house

        def rhs(t_s, y):
            T_roof, T_attic, T_in = y
            w = wx(t_s / 3600.0)
            f = self.roof_fluxes(T_roof, T_attic, w['T_amb'], w['ghi'], w['humidity'], w['windspeed'], w['cloud_cover'])
            dT_roof = sum(f.values()) / h.roof_heat_capacity
            q_ceiling = h.K_ceiling * (T_attic - T_in)
            dT_attic = (h.K_roof_attic * (T_roof - T_attic) + h.K_attic_vent * (w['T_amb'] - T_attic)
                        - q_ceiling) / h.C_attic
            dT_in = (q_ceiling + h.K_ambient * (w['T_amb'] - T_in) + h.gains(w['ghi'])) / h.C_in
            return [dT_roof, dT_attic, dT_in]

        if T0 is None:
            T_amb0 = wx(t_h[0])['T_amb']
            T0 = [T_amb0, T_amb0, T_amb0]
        t_start = (t_h[0] - spinup_days * period) * 3600.0
        t_eval = t_h * 3600.0
        sol = solve_ivp(rhs, (t_start, t_eval[-1]), T0, t_eval=t_eval, method='LSODA',
                        rtol=1e-4, atol=1e-3, max_step=900.0)
        if not sol.success:
            raise RuntimeError(sol.message)
        out = weather.copy()
        out['T_roof'], out['T_attic'], out['T_in'] = sol.y
        out['ceiling_heat_into_house_W'] = h.K_ceiling * (out['T_attic'] - out['T_in'])
        return out


def daily_weather(T_max=32.0, daily_range=10.0, ghi_peak=1000.0, humidity=50.0, windspeed=2.0,
                  cloud_cover=0.0, sunrise=6.0, sunset=18.0, hour_of_T_max=15.0, step_minutes=15):
    """
    An idealised clear day: sinusoidal sunshine between sunrise and sunset, and air temperature
    peaking at `hour_of_T_max` with a cosine diurnal swing of `daily_range`.
    Returns a DataFrame indexed by hour of day (0-24).
    """
    hours = np.arange(0, 24 + 1e-9, step_minutes / 60.0)
    day_len = sunset - sunrise
    ghi = ghi_peak * np.clip(np.sin(np.pi * (hours - sunrise) / day_len), 0, None)
    ghi[(hours < sunrise) | (hours > sunset)] = 0.0
    T_amb = T_max - daily_range / 2 * (1 - np.cos(2 * np.pi * (hours - hour_of_T_max) / 24))
    return pd.DataFrame(dict(T_amb=T_amb, ghi=ghi, humidity=humidity, windspeed=windspeed,
                             cloud_cover=cloud_cover), index=pd.Index(hours, name='hour'))


def compare_roofs(paints=('charcoal', 'cool'), house=None, mode='day', **weather):
    """
    Convenience for the demo: same house, same weather, different roofs.
    mode='steady' -> dict of steady_state() results per paint (weather: T_amb, ghi, humidity, windspeed, cloud_cover)
    mode='day'    -> dict of simulate() DataFrames per paint (weather: kwargs of daily_weather)
    """
    house = house if house is not None else House()
    out = {}
    if mode == 'steady':
        for p in paints:
            out[p] = RoofedHouse(house, paint=p).steady_state(**weather)
    else:
        wx = daily_weather(**weather)
        for p in paints:
            out[p] = RoofedHouse(house, paint=p).simulate(wx)
    return out
