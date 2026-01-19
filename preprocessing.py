import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

try:
    photometry_data_path = "ceers_all_v0.51_eazy.hdf5"
    photometry_data_container = Table.read(photometry_data_path)
    # print(photometry_data_container.columns)

except Exception as error_1:
    print(f" { type(error_1).__name__ } ocured! ")



def processed_data():
    
    z_spec = photometry_data_container['z_spec']
    log_mass = photometry_data_container['fast_lmass']
    flux_115_magnitude = -2.5*np.log10(photometry_data_container['FLUX_115'])+31.4 # had conversion aid from Prof McGrath
    flux_150_magnitude = -2.5*np.log10(photometry_data_container['FLUX_150'])+31.4
    flux_200_magnitude = -2.5*np.log10(photometry_data_container['FLUX_200'])+31.4
    flux_277_magnitude = -2.5*np.log10(photometry_data_container['FLUX_277'])+31.4
    flux_356_magnitude = -2.5*np.log10(photometry_data_container['FLUX_356'])+31.4
    flux_410_magnitude = -2.5*np.log10(photometry_data_container['FLUX_410'])+31.4
    flux_444_magnitude = -2.5*np.log10(photometry_data_container['FLUX_444'])+31.4
    flux_606_magnitude = -2.5*np.log10(photometry_data_container['FLUX_606'])+31.4
    flux_814_magnitude = -2.5*np.log10(photometry_data_container['FLUX_814'])+31.4
    flux_125_magnitude = -2.5*np.log10(photometry_data_container['FLUX_125'])+31.4


    # we perform the first series of feature extraction here based on the brightness of the galaxy using flux_356 < 26.5
    flux_356_magnitude = -2.5*np.log10(photometry_data_container['FLUX_356'])+31.4
    mask_from_flux = flux_356_magnitude < 26.5 # this is for the first round of data cleaning.
    # print(mask_for_flux)
    # print(flux_356_magnitude)

    new_z_spec = z_spec[mask_from_flux]
    new_log_mass = log_mass[mask_from_flux]
    new_flux_115_magnitude = flux_115_magnitude[mask_from_flux]
    new_flux_150_magnitude = flux_150_magnitude[mask_from_flux]
    new_flux_200_magnitude = flux_200_magnitude[mask_from_flux]
    new_flux_277_magnitude = flux_277_magnitude[mask_from_flux]
    new_flux_356_magnitude = flux_356_magnitude[mask_from_flux]
    new_flux_410_magnitude = flux_410_magnitude[mask_from_flux]
    new_flux_444_magnitude = flux_444_magnitude[mask_from_flux]
    new_flux_606_magnitude = flux_606_magnitude[mask_from_flux]
    new_flux_814_magnitude = flux_814_magnitude[mask_from_flux]
    new_flux_125_magnitude = flux_125_magnitude[mask_from_flux]


    # now, we perfomr the remainder of the feature extraction based on the z_spec, extracting only features for galaxies with z_spec not equal to -99 or -1.
    # the first feature we'd want to extract is the true spectroscopic redshift z_spec from which we'll discard galaxies with redshits equal to -99 or -1
 
    mask_from_z_spec = ~( (new_z_spec == -99) | (new_z_spec == -1) )

    desired_z_spec = new_z_spec[mask_from_z_spec]
    desired_log_mass = new_log_mass[mask_from_z_spec]
    desired_flux_115_magnitude = new_flux_115_magnitude[mask_from_z_spec]
    desired_flux_150_magnitude = new_flux_150_magnitude[mask_from_z_spec]
    desired_flux_200_magnitude = new_flux_200_magnitude[mask_from_z_spec]
    desired_flux_277_magnitude = new_flux_277_magnitude[mask_from_z_spec]
    desired_flux_356_magnitude = new_flux_356_magnitude[mask_from_z_spec]
    desired_flux_410_magnitude = new_flux_410_magnitude[mask_from_z_spec]
    desired_flux_444_magnitude = new_flux_444_magnitude[mask_from_z_spec]
    desired_flux_606_magnitude = new_flux_606_magnitude[mask_from_z_spec]
    desired_flux_814_magnitude = new_flux_814_magnitude[mask_from_z_spec]
    desired_flux_125_magnitude = new_flux_125_magnitude[mask_from_z_spec]


    desired_z_phot = (photometry_data_container['z_phot'][mask_from_flux])[mask_from_z_spec]

    X = np.vstack([desired_log_mass, desired_flux_115_magnitude, desired_flux_150_magnitude, desired_flux_200_magnitude, desired_flux_277_magnitude, desired_flux_356_magnitude, desired_flux_410_magnitude, desired_flux_444_magnitude, desired_flux_606_magnitude, desired_flux_814_magnitude, desired_flux_125_magnitude]).T
    # print(X[:,0].shape)
    y = np.array(desired_z_spec)
    # print(f" The shape of desired_z_spec is {desired_z_spec.shape}")
    # print(f" The shape of desired_z_phot is {desired_z_phot.shape}")
    # print(X)
    # print(y)
    return X, y




if __name__ == "__main__":
    processed_data()
    # X, y = processed_data()
    # print(y)

