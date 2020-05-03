import numpy as np
from physics.hydro.state import HydroState, FloatType


class MhdState(HydroState):
    def __init__(self,
                 mass=None,
                 density=None,
                 velocity=None,
                 specific_thermal_energy=None,
                 magnetic_field=None,
                 lorentz_heaviside=True,
                 gamma: FloatType = FloatType(5.0 / 3.0),
                 unit_length_in_cm: FloatType = FloatType(1.0),
                 unit_mass_in_g: FloatType = FloatType(1.0),
                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                 hubble_param: FloatType = FloatType(1.0),
                 scale_factor: FloatType = FloatType(1.0)):
        super().__init__(
            mass=mass,
            density=density,
            velocity=velocity,
            specific_thermal_energy=specific_thermal_energy,
            gamma=gamma,
            unit_length_in_cm=unit_length_in_cm,
            unit_mass_in_g=unit_mass_in_g,
            unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s,
            hubble_param=hubble_param,
            scale_factor=scale_factor)
        self._magnetic_field = np.array(magnetic_field, dtype=FloatType, ndmin=2)
        self._lorentz_heaviside = lorentz_heaviside

    """ factories """

    @staticmethod
    def from_conserved_variables(volume=None,
                                 mass=None,
                                 momentum=None,
                                 thermal_energy=None,
                                 conserved_magnetic_field=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)):
        """ create MhdState from conserved variables """
        _mass = np.array(mass, dtype=FloatType, ndmin=1)
        _momentum = np.array(momentum, dtype=FloatType, ndmin=2)
        density = _mass / np.array(volume, dtype=FloatType, ndmin=1)
        velocity = np.zeros(_momentum.shape)
        for dim in np.arange(_momentum.shape[1]):
            velocity[:, dim] = _momentum[:, dim] / _mass
        specific_thermal_energy = thermal_energy / _mass
        magnetic_field = np.array(conserved_magnetic_field) / np.array(volume)
        return MhdState(mass=_mass, density=density, velocity=velocity,
                        specific_thermal_energy=specific_thermal_energy,
                        magnetic_field=magnetic_field, gamma=gamma,
                        unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                        unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                        scale_factor=scale_factor)

    @staticmethod
    def from_primitive_variables(volume=None,
                                 density=None,
                                 velocity=None,
                                 specific_thermal_energy=None,
                                 magnetic_field=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)):
        """ create MhdState from primitive variables """
        mass = density * volume
        return MhdState(mass=mass, density=density, velocity=velocity,
                        specific_thermal_energy=specific_thermal_energy,
                        magnetic_field=magnetic_field, gamma=gamma,
                        unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                        unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                        scale_factor=scale_factor)

    @staticmethod
    def from_arepo_snapshot(snap):
        """
        direct interface from inspector gadget library
        :param snap: arepo.Simulation object (inspector gadget library)
        :return: MhdState object
        """
        scale_factor = 1.0
        if snap.parameters.ComovingIntegrationOn == 1:
            scale_factor = snap.time
        return MhdState(
            mass=np.array(snap.part0.mass, dtype=FloatType),
            density=np.array(snap.part0.rho, dtype=FloatType),
            velocity=np.array(snap.part0.velocity, dtype=FloatType),
            specific_thermal_energy=np.array(snap.part0.u, dtype=FloatType),
            magnetic_field=np.array(snap.MagneticField, dtype=FloatType),
            gamma=FloatType(5.0 / 3.0),
            unit_length_in_cm=FloatType(snap.header.UnitLength_in_cm),
            unit_mass_in_g=FloatType(snap.header.UnitMass_in_g),
            unit_velocity_in_cm_per_s=FloatType(snap.header.UnitVelocity_in_cm_per_s),
            hubble_param=FloatType(snap.header.HubbleParam),
            scale_factor=FloatType(scale_factor))

    """ magic methods """

    def __str__(self):
        out = super().__str__()
        out += "\n magnetic field = " + str(self._magnetic_field)
        return out

    def __repr__(self):
        out = super().__repr__()[:-2]
        out += "; magnetic_field = " + str(self._magnetic_field) + " >"
        return out

    def __eq__(self, other):
        if np.all(self.mass == other.mass) \
                and np.all(self.density == other.density) \
                and np.all(self.velocity == other.velocity) \
                and np.all(self.specific_thermal_energy == other.specific_thermal_energy) \
                and np.all(self.magnetic_field == other.magnetic_field):
            return True
        return False

    """ properties (proper gauss cgs units!)"""

    @property
    def magnetic_field(self):
        unit_magnetic_field = np.sqrt(self._unit_mass_in_g) * np.power(self._unit_length_in_cm, -1.5)
        unit_magnetic_field *= self._unit_velocity_in_cm_per_s
        if self._lorentz_heaviside:
            unit_magnetic_field *= np.sqrt(4.0 * np.pi)
        # cosmological factor
        unit_magnetic_field *= self._hubble_param / self._scale_factor / self._scale_factor

        return self._magnetic_field * unit_magnetic_field

    @property
    def magnetic_energy_density(self):
        egy = self.magnetic_field[:, 0] ** 2 + self.magnetic_field[:, 1] ** 2 + self.magnetic_field[:, 2] ** 2
        egy /= 8.0 * np.pi
        return egy

    @property
    def magnetic_energy(self):
        return self.magnetic_energy_density * self.volume

    @property
    def pressure(self):
        return self.thermal_pressure + self.magnetic_energy_density

    @property
    def alfven_velocity_squared(self):
        va2 = self.magnetic_field[:, 0] ** 2 + self.magnetic_field[:, 1] ** 2 + self.magnetic_field[:, 2] ** 2
        va2 /= 4.0 * np.pi * self.density
        return va2

    @property
    def alfven_velocity(self):
        return np.sqrt(self.alfven_velocity_squared)

    @property
    def sound_speed_squared(self):
        cs2 = self._gamma * (self._gamma - 1.0) * self._specific_thermal_energy * self._unit_velocity_in_cm_per_s ** 2
        cs2 += self.alfven_velocity_squared
        return cs2

    @property
    def alfvenic_mach_number(self):
        return self.absolute_velocity / self.alfven_velocity

    @property
    def total_energy(self):
        return super().total_energy + self.magnetic_energy

    def projected_magnetic_field(self, direction):
        """

        :param direction: np.array
            unit vector in which magnetic field should be projected
        :return:
        """
        return np.inner(self.magnetic_field, direction)
