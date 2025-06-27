import numpy as np
import physics.constants as constants

FloatType = np.float64


class HydroState(object):
    """
    class for hydrodynamic state
    """

    def __init__(self,
                 mass=None,
                 density=None,
                 velocity=None,
                 specific_thermal_energy=None,
                 gamma: FloatType = FloatType(5.0 / 3.0),
                 unit_length_in_cm: FloatType = FloatType(1.0),
                 unit_mass_in_g: FloatType = FloatType(1.0),
                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                 hubble_param: FloatType = FloatType(1.0),
                 scale_factor: FloatType = FloatType(1.0)
                 ):
        self._mass = np.array(mass, dtype=FloatType, ndmin=1)
        self._density = np.array(density, dtype=FloatType, ndmin=1)
        self._velocity = np.array(velocity, dtype=FloatType, ndmin=2)
        self._specific_thermal_energy = np.array(specific_thermal_energy, dtype=FloatType, ndmin=1)
        self._gamma = FloatType(gamma)
        self._unit_length_in_cm = FloatType(unit_length_in_cm)
        self._unit_mass_in_g = FloatType(unit_mass_in_g)
        self._unit_velocity_in_cm_per_s = FloatType(unit_velocity_in_cm_per_s)
        self._hubble_param = FloatType(hubble_param)
        self._scale_factor = FloatType(scale_factor)

    """ factories """

    @staticmethod
    def from_conserved_variables(volume=None,
                                 mass=None,
                                 momentum=None,
                                 thermal_energy=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)
                                 ):
        """ create HydroState from conserved variables """
        _mass = np.array(mass, dtype=FloatType, ndmin=1)
        _momentum = np.array(momentum, dtype=FloatType, ndmin=2)
        density = _mass / np.array(volume, dtype=FloatType, ndmin=1)
        velocity = np.zeros(_momentum.shape)
        for dim in np.arange(_momentum.shape[1]):
            velocity[:, dim] = _momentum[:, dim] / _mass
        specific_thermal_energy = thermal_energy / _mass
        return HydroState(mass=_mass, density=density, velocity=velocity,
                          specific_thermal_energy=specific_thermal_energy, gamma=gamma,
                          unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                          unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                          scale_factor=scale_factor)

    @staticmethod
    def from_primitive_variables(volume=None,
                                 density=None,
                                 velocity=None,
                                 specific_thermal_energy=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)
                                 ):
        """ create HydroState from primitive variables """
        mass = density * volume
        return HydroState(mass=mass, density=density, velocity=velocity,
                          specific_thermal_energy=specific_thermal_energy, gamma=gamma,
                          unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                          unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                          scale_factor=scale_factor)

    @staticmethod
    def from_arepo_snapshot(snap, center_of_mass_velocity=np.array([0,0,0], dtype=FloatType)):
        """
        direct interface from inspector gadget library
        :param snap: arepo.Simulation object (inspector gadget library)
        :return: HydroState object
        """
        scale_factor = 1.0
        if snap.parameters.ComovingIntegrationOn == 1:
            scale_factor = snap.time
        
        vel = np.array(snap.part0.velocity, dtype=FloatType)
        vel[:,0] -= center_of_mass_velocity[0]
        vel[:,1] -= center_of_mass_velocity[1]
        vel[:,2] -= center_of_mass_velocity[2]

        return HydroState(
            mass=np.array(snap.part0.mass, dtype=FloatType),
            density=np.array(snap.part0.rho, dtype=FloatType),
            velocity=vel,
            specific_thermal_energy=np.array(snap.part0.u, dtype=FloatType),
            gamma=FloatType(5.0 / 3.0),
            unit_length_in_cm=FloatType(snap.header.UnitLength_in_cm),
            unit_mass_in_g=FloatType(snap.header.UnitMass_in_g),
            unit_velocity_in_cm_per_s=FloatType(snap.header.UnitVelocity_in_cm_per_s),
            hubble_param=FloatType(snap.header.HubbleParam),
            scale_factor=FloatType(scale_factor)
        )

    """ magic methods """

    def __str__(self):
        out = "hydro state:"
        out += "\n mass = " + str(self._mass)
        out += "\n density = " + str(self._density)
        out += "\n velocity = " + str(self._velocity)
        out += "\n specific thermal energy = " + str(self._specific_thermal_energy)
        return out

    def __repr__(self):
        out = "hydro state: < "
        out += "mass = " + str(self._mass)
        out += "; density = " + str(self._density)
        out += "; velocity = " + str(self._velocity)
        out += "; specific thermal energy = " + str(self._specific_thermal_energy) + " >"
        return out

    def __eq__(self, other):
        if np.all(self.mass == other.mass) \
                and np.all(self.density == other.density) \
                and np.all(self.velocity == other.velocity) \
                and np.all(self.specific_thermal_energy == other.specific_thermal_energy):
            return True
        return False

    """ properties (proper cgs units!)"""

    @property
    def mass(self):
        return self._mass * self._unit_mass_in_g / self._hubble_param

    @property
    def density(self):
        unit_density = self._unit_mass_in_g / self._unit_length_in_cm / self._unit_length_in_cm / self._unit_length_in_cm
        unit_density *= self._hubble_param * self._hubble_param / self._scale_factor / self._scale_factor / self._scale_factor
        return self._density * unit_density

    @property
    def volume(self):
        unit_volume = self._unit_length_in_cm * self._unit_length_in_cm * self._unit_length_in_cm
        unit_volume *= self._scale_factor * self._scale_factor * self._scale_factor
        unit_volume /= self._hubble_param * self._hubble_param * self._hubble_param
        return self._mass / self._density * unit_volume

    @property
    def velocity(self):
        return self._velocity * self._unit_velocity_in_cm_per_s * np.sqrt(self._scale_factor)

    @property
    def absolute_velocity(self):
        vel = np.array(self.velocity, dtype=FloatType, ndmin=2)
        abs_vel = np.zeros(vel.shape[0], dtype=FloatType)
        for i in np.arange(vel.shape[1]):
            abs_vel[:] += vel[:, i] * vel[:, i]
        return np.sqrt(abs_vel)

    @property
    def momentum(self):
        unit_momentum = self._unit_velocity_in_cm_per_s * self._unit_mass_in_g
        unit_momentum /= self._hubble_param * self._hubble_param / np.sqrt(self._scale_factor)
        return self._velocity * self._mass * unit_momentum

    @property
    def kinetic_energy(self):
        unit_energy = self._unit_mass_in_g * self._unit_velocity_in_cm_per_s ** 2
        unit_energy *= self._scale_factor / self._hubble_param
        v_squared = (self._velocity[:, 0] ** 2 + self._velocity[:, 1] ** 2 + self._velocity[:, 2] ** 2)
        return 0.5 * self._mass * v_squared * unit_energy

    @property
    def specific_thermal_energy(self):
        unit_spec_energy = self._unit_velocity_in_cm_per_s ** 2
        return self._specific_thermal_energy * unit_spec_energy

    @property
    def thermal_energy_density(self):
        unit_energy_density = self._unit_mass_in_g * self._unit_velocity_in_cm_per_s ** 2
        unit_energy_density /= self._unit_length_in_cm ** 3
        unit_energy_density *= self._hubble_param ** 2 / self._scale_factor ** 2
        return self._specific_thermal_energy * self._density * unit_energy_density

    @property
    def thermal_pressure(self):
        unit_pressure = self._unit_mass_in_g * self._unit_velocity_in_cm_per_s ** 2
        unit_pressure /= self._unit_length_in_cm ** 3
        unit_pressure *= self._hubble_param ** 2 / self._scale_factor ** 2
        return (self._gamma - 1.0) * self._density * self._specific_thermal_energy * unit_pressure

    @property
    def pressure(self):
        return self.thermal_pressure

    @property
    def thermal_energy(self):
        unit_energy = self._unit_mass_in_g * self._unit_velocity_in_cm_per_s ** 2
        unit_energy *= self._scale_factor / self._hubble_param
        return self._specific_thermal_energy * self._mass * unit_energy

    @property
    def sound_speed_squared(self):
        return self._gamma * (self._gamma - 1.0) * self._specific_thermal_energy * self._unit_velocity_in_cm_per_s ** 2

    @property
    def sound_speed(self):
        return np.sqrt(self.sound_speed_squared)

    @property
    def mach_number(self):
        return self.absolute_velocity / self.sound_speed

    @property
    def total_energy(self):
        return self.thermal_energy + self.kinetic_energy

    """ functions that use external information """

    def cooling_luminosity(self, cooling_rate, hydrogen_mass_fraction=0.76):
        """

        :param cooling_rate: np.array()
            cooling rate Lambda/n_h^2 in cgs units
        :param hydrogen_mass_fraction:
            hydrogen mass fraction of cells
        :return: np.array
            cooling luminosity in erg/s
        """
        hydrogen_number_density = self.density * hydrogen_mass_fraction / constants.PROTONMASS
        return cooling_rate * hydrogen_number_density * hydrogen_number_density * self.volume

    def temperature(self, mu=None, hydrogen_fraction=0.76, electron_abundance=1.16):
        """

        :param mu: (optional)
            mean molecular weight;  will override hydrogen fraction and electron abundance if set
        :param hydrogen_fraction:
            mass fraction in hydrogen to calculate mean molecular weight
        :param electron_abundance:
            X_e / X_h number of electrons per hydogen atom
        :return:
            array with temperatures
        """
        if isinstance(mu, type(None)):
            mu = 4.0 / (1.0 + 3.0 * hydrogen_fraction + 4.0 * hydrogen_fraction * electron_abundance)
            print(f'mu = {mu}')

        return (self._gamma - 1.0) * self.specific_thermal_energy / constants.BOLTZMANN * mu * constants.PROTONMASS

    def hydrogen_number_density(self, hydrogen_fraction=0.76):
        """

        :param hydrogen_fraction:
            mass fraction in hydrogen
        :return:
            array with hydrogen number densities
        """
        return np.array(hydrogen_fraction * self.density / constants.PROTONMASS, dtype=FloatType)

    def electron_number_density(self, hydrogen_fraction=0.76, electron_abundance=1.16):
        """

        :param hydrogen_fraction:
            mass fraction in hydrogen
        :param electron_abundance:
            X_e / X_h number of electrons per hydogen atom
        :return:
            array with electron number densities
        """
        return np.array(electron_abundance * self.hydrogen_number_density(hydrogen_fraction=hydrogen_fraction),
                        dtype=FloatType)

    def pseudo_entropy(self, hydrogen_fraction=0.76, electron_abundance=1.16):
        """

        :param hydrogen_fraction:
            mass fraction in hydrogen
        :param electron_abundance:
            X_e / X_h number of electrons per hydogen atom
        :return:
            pseudo entropy T * n_e^(-2/3)
        """
        return self.temperature(hydrogen_fraction=hydrogen_fraction,
                                electron_abundance=electron_abundance) * self.electron_number_density(
            hydrogen_fraction=hydrogen_fraction, electron_abundance=electron_abundance) ** (-2. / 3.)
