import GCRCatalogs
import numpy as np
from astropy.table import Table
from GCR import GCRQuery
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def nike () :
    cats=GCRCatalogs.get_available_catalogs()

    gc = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_small')
    cols=gc.list_all_quantities()
    scols= []
    scols.append([x for x in cols if 
              (not x.startswith("sed")) and 
              (not x.endswith("no_host_extinction")) and 
              (not x.startswith("ellipticity")) and
              (not x.startswith("size"))
             ])

    galaxy_data = gc.get_quantities(['ra', 'dec', "redshift", "stellar_mass", 
                                 "mag_u_lsst", "mag_g_lsst", "mag_r_lsst", "mag_i_lsst", "mag_z_lsst",
                                 'halo_id', 'halo_mass', 'is_central'], filters=['mag_r < 23', 'Mag_true_r_lsst_z0 < -19.0'])

    cluster_data = gc.get_quantities(['ra','dec', "redshift", 'stellar_mass', 'halo_mass', 'halo_id'], 
                                 filters=['is_central', 'halo_mass > 1e14', 'redshift >= 0.3', 'redshift < 0.6'])
    cluster_data = pd.DataFrame(cluster_data)
    cluster_data["central_sm"] = cluster_data["stellar_mass"]
    cluster_data = cluster_data.drop(columns="stellar_mass")


    print("galaxies", len(galaxy_data), type(galaxy_data), galaxy_data["ra"].size)
    print("clusters", len(cluster_data), type(cluster_data), cluster_data["ra"].size)
    print("cluster masses", np.log10(cluster_data["halo_mass"].min()),np.log10(cluster_data["halo_mass"].max()))



    radius_arcmin=7.
    gal_df = all_cutout_around_cluster(cluster_data, galaxy_data, radius_arcmin)

    print("gal_df:  ", gal_df.size)


    df=stellarmass_to_cat(cluster_data, galaxy_data)
    print("df:  ", gal_df.size)



    cluster_data["halo_mass"] = np.log10(cluster_data["halo_mass"])
    cluster_data["stellarmass"] = np.log10(cluster_data["stellarmass"])
    cluster_data["central_sm"] = np.log10(cluster_data["central_sm"])
    cluster_data["sm_0.50"] = np.log10(cluster_data["sm_0.50"])
    cluster_data["sm_0.67"] = np.log10(cluster_data["sm_0.67"])
    cluster_data["sm_1.0"] = np.log10(cluster_data["sm_1.0"])
    cluster_data["sm_1.5"] = np.log10(cluster_data["sm_1.5"])
    cluster_data["sm_2.0"] = np.log10(cluster_data["sm_2.0"])
    cluster_data["sm_3.0"] = np.log10(cluster_data["sm_3.0"])


    gal_df.to_csv("~/Data/galaxies_near_clusters_0.3-0.6.csv") 
    #cluster_data.to_csv("~/Data/cluster_data_0.3-0.6.csv")


def cutout_around_cluster (galaxy_data, radius_arcmin, ra,dec, halo_id, halo_mass, halo_z) :
    radius=radius_arcmin/60.
    ix, = np.where((np.abs(galaxy_data["dec"]-dec)<radius) & 
                 (np.abs(galaxy_data["ra"]-ra)<radius/np.cos(dec*2*np.pi/360.)) )
    new_df = pd.DataFrame(
            {"ra":galaxy_data["ra"][ix],"dec":galaxy_data["dec"][ix],
             "redshift":galaxy_data["redshift"][ix],
             "stellar_mass":np.log10(galaxy_data["stellar_mass"][ix]),
             "mag_i":galaxy_data["mag_i_lsst"][ix],
             "mag_u-g":galaxy_data["mag_u_lsst"][ix]-galaxy_data["mag_g_lsst"][ix],
             "mag_g-r":galaxy_data["mag_g_lsst"][ix]-galaxy_data["mag_r_lsst"][ix],
             "mag_r-i":galaxy_data["mag_r_lsst"][ix]-galaxy_data["mag_i_lsst"][ix],
             "mag_i-z":galaxy_data["mag_i_lsst"][ix]-galaxy_data["mag_z_lsst"][ix],
             "halo_id":galaxy_data["halo_id"][ix],
             "cluster_id":(np.ones(galaxy_data["halo_id"][ix].size)*halo_id).astype(int),
             "cluster_mass":np.log10((np.ones(galaxy_data["halo_id"][ix].size)*halo_mass)),
             "cluster_z":(np.ones(galaxy_data["halo_id"][ix].size)*halo_z),
             "is_central":galaxy_data["is_central"][ix],
            })
    cl_radius_arcmin = np.sqrt((new_df["dec"]-dec)**2 + ((new_df["ra"]-ra)*np.cos(dec*2*np.pi/360.))**2)*60.
    new_df["cluster_radius_arcmin"] = cl_radius_arcmin
    return new_df



def all_cutout_around_cluster (cluster_data, galaxy_data, radius_arcmin) :
    list=[]
    for cl_idx in range(0,cluster_data["ra"].size): 
        ra = cluster_data["ra"].iloc[cl_idx]
        dec = cluster_data["dec"].iloc[cl_idx];
        halo_id = cluster_data["halo_id"].iloc[cl_idx]
        halo_mass = cluster_data["halo_mass"].iloc[cl_idx] 
        halo_z = cluster_data["redshift"].iloc[cl_idx]
        list.append(cutout_around_cluster (galaxy_data, radius_arcmin, ra,dec, halo_id, halo_mass, halo_z ))
    gal_df = pd.concat(list, ignore_index=True) # concat all cluster df into one single gal_cat df
    return gal_df


# given raw galaxy_data and cluster_data, add a series of stellar masses to cluster_data
def stellarmass_to_cat(cluster_data, galaxy_data) :
    rad_list = np.array( [0.5, 0.67, 1.0, 1.5, 2, 3, 10]) # mpc
    name_list = ["sm_0.50","sm_0.67", "sm_1.0", "sm_1.5", "sm_2.0", "sm_3.0", "stellarmass"]
    rad_list = rad_list*3.33 # arcmin/mpc
    if (len(rad_list) != len(name_list)) : 
        raise Exception("radius and names !same length")    
    for radius_arcmin, name in zip(rad_list,name_list) :
        print(name, radius_arcmin)
        sm = []
        for i in range(0,cluster_data["halo_id"].size) :
            ra=cluster_data["ra"].iloc[i]; 
            dec=cluster_data["dec"].iloc[i]; 
            halo_id = cluster_data["halo_id"].iloc[i]
            halo_mass = cluster_data["halo_mass"].iloc[i];  
            halo_z = cluster_data["redshift"].iloc[i]
            df=cutout_around_cluster (galaxy_data, radius_arcmin, ra,dec, halo_id, halo_mass, halo_z) 
            match_df = df[df["halo_id"] == df["cluster_id"]]
            stellarmass = match_df["stellar_mass"].sum()
            sm.append(stellarmass)
        sm=np.array(sm)
        cluster_data[name] = sm
    return cluster_data
