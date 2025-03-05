# import torch
import numpy as np
import sys

sys.path.append("../")
# from configs.base_config import TrainingConfig
# from utils import get_dataloader, TransformFields
import matplotlib.pyplot as plt

# import einops
# from torch.nn.functional import interpolate
from glob import glob
import xarray as xr
import numpy as np


def get_deptht(restart, mask):
    """
    Calculate the depth of each vertical level on grid T in the 3D grid.

    Parameters:
        restart (xarray.Dataset)     : The dataset containing ocean model variables.
        mask (xarray.Dataset) : The dataset containing mask variables.

    Returns:
        deptht (numpy.array) : The depth of each vertical level.
    """
    ssh = restart.sshn.squeeze()
    e3w_0 = (
        mask.e3w_0.squeeze()
    )  # initial z axis cell's thickness on grid W - (t,z,y,x)
    e3t_0 = (
        mask.e3t_0.squeeze()
    )  # initial z axis cell's thickness on grid T - (t,z,y,x)
    tmask = (
        mask.tmask.squeeze()
    )  # grid T continent mask                     - (t,z,y,x)
    ssmask = tmask[:, 0]  # bathymetry                                - (t,y,x)
    bathy = e3t_0.sum(
        dim="depth"
    )  # initial condition depth 0                 - (t,z,y,x)
    depth_0 = e3w_0.copy().squeeze()
    depth_0[:, 0] = 0.5 * e3w_0[:, 0]
    depth_0[:, 1:] = depth_0[:, 0:1].data + e3w_0[:, 1:].cumsum(dim="depth")
    deptht = depth_0 * (1 + ssh / (bathy + 1 - ssmask)) * tmask
    return deptht


def get_density(thetao, so, depth, tmask):
    """
    Compute potential density referenced at the surface and density anomaly.

    Parameters:
        thetao (numpy.array) : Temperature array - (t,z,y,x).
        so (numpy.array)     : Salinity array    - (t,z,y,x).
        depth (numpy.array)  : Depth array       - (t,z,y,x).
        tmask (numpy.array)  : Mask array        - (t,z,y,x).

    Returns:
        tuple: A tuple containing:
            array: Potential density referenced at the surface.
            array: Density anomaly.
    """
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0
    r1_Z0 = 1.0e-4

    EOS000 = 8.0189615746e02
    EOS100 = 8.6672408165e02
    EOS200 = -1.7864682637e03
    EOS300 = 2.0375295546e03
    EOS400 = -1.2849161071e03
    EOS500 = 4.3227585684e02
    EOS600 = -6.0579916612e01
    EOS010 = 2.6010145068e01
    EOS110 = -6.5281885265e01
    EOS210 = 8.1770425108e01
    EOS310 = -5.6888046321e01
    EOS410 = 1.7681814114e01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e01
    EOS120 = 6.1548258127e01
    EOS220 = -6.0362551501e01
    EOS320 = 2.9130021253e01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e01
    EOS130 = -3.3449108469e01
    EOS230 = 1.9717078466e01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e01
    EOS101 = -4.2549998214e01
    EOS201 = 5.0774768218e01
    EOS301 = -3.0938076334e01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01

    zh = depth * r1_Z0  # depth
    zt = thetao * r1_T0  # temperature
    zs = np.sqrt(np.abs(so + rdeltaS) * r1_S0)  # square root salinity
    ztm = tmask.squeeze()

    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    zn2 = (
        (EOS022 * zt + EOS112 * zs + EOS012) * zt + (EOS202 * zs + EOS102) * zs + EOS002
    )
    zn1 = (
        (
            (
                (EOS041 * zt + EOS131 * zs + EOS031) * zt
                + (EOS221 * zs + EOS121) * zs
                + EOS021
            )
            * zt
            + ((EOS311 * zs + EOS211) * zs + EOS111) * zs
            + EOS011
        )
        * zt
        + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs
        + EOS001
    )
    zn0 = (
        (
            (
                (
                    (
                        (EOS060 * zt + EOS150 * zs + EOS050) * zt
                        + (EOS240 * zs + EOS140) * zs
                        + EOS040
                    )
                    * zt
                    + ((EOS330 * zs + EOS230) * zs + EOS130) * zs
                    + EOS030
                )
                * zt
                + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs
                + EOS020
            )
            * zt
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110)
            * zs
            + EOS010
        )
        * zt
        + (
            ((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs
            + EOS100
        )
        * zs
        + EOS000
    )

    zn = ((zn3 * zh + zn2) * zh + zn1) * zh + zn0

    rhop = zn0 * ztm  # potential density referenced at the surface
    rho_insitu = zn * ztm  # density anomaly (masked)
    return rhop, rho_insitu


def check_density(density, epsilon=1e-5):
    """
    args :
        density (xarray) : DataArray (t, depth, lat, lon) with density value for each point of the grid.
    return :
        (float) proportion of points not respecting density decreasing constraint
    """
    density = density.where(density != 0)
    diff = density - density.shift(depth=-1)
    return (
        (diff > epsilon).mean().data
    )  # Proportion of points not respecting decreasing density


def temperature_500m_30NS_metric(temperature, file_mask):
    """
    Metric Extraction Function :
    Average Temperature at 500m depth between 30N and 30S.
    Unit : °C


    Input :
       -  thetao    : xarray.DataArray
       -  file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files

    """

    # Taking Temperature At 500m depth and between 30N and 30S.

    t500_30NS = temperature.sel(depth=500, method="nearest").where(
        abs(temperature.nav_lat) < 30, drop=False
    )

    # Computing Area Weights from Mask over 30N-30S latitude zone and @500m depth
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    tmask = file_mask.tmask.squeeze()
    area_500m_30NS = (
        e1t
        * e2t
        * tmask.sel(depth=500, method="nearest").where(
            abs(temperature.nav_lat) < 30, drop=False
        )
    )

    # Returning Average Temperature at 500m depth as a numpy scalar
    return (t500_30NS * area_500m_30NS).sum(
        dim=["nav_lat", "nav_lon"]
    ) / area_500m_30NS.sum(dim=["nav_lat", "nav_lon"])


def temperature_BWbox_metric(thetao, file_mask):
    """
    Metric Extraction Function :
    Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below 3000m or beyond 30 degrees of latitude North and South.

    ________________________________________________ _Surface
    | . . . . |__________________________| . . . . |_500m
    | . . . . |                          | . . . . |
    | . . . . |        Deep Water        | . . . . |
    | . . . . |__________________________| . . . . |_3000m
    | . . . . . . . . Bottom Water . . . . . . . . |
    |______________________________________________|_Bottom
    S        30S           Eq.          30N        N

    Figure : Schematic Representation of the Bottom Water box used in this metric.

    Unit : °C

    Input :
       -  thetao    : xarray.DataArray
       -  file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files

    """

    t_BW = thetao.where(1 - (thetao.depth < 3000) * (abs(thetao.nav_lat) < 30))

    # Computing Area Weights from Mask over Box
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    tmask = file_mask.tmask.squeeze()
    area_BW = (
        e1t * e2t * tmask.where(1 - (thetao.depth < 3000) * (abs(thetao.nav_lat) < 30))
    )

    # Returning Average Temperature on Box
    return (t_BW * area_BW).sum(dim=["nav_lat", "nav_lon", "depth"]) / area_BW.sum(
        dim=["nav_lat", "nav_lon", "depth"]
    )


def ACC_Drake_metric(uo, file_mask):
    """
    Metric Extraction Function :
    Antarctic Circumpolar Current Transport at the DINO equivalent of the Drake Passage (x=0).
    Unit : Sv


    Input :
       -    uo        : xarray.DataArray
       -    file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files


    """

    umask_Drake = file_mask.umask.isel(nav_lon=0).squeeze()
    e3u = file_mask.e3u_0.squeeze()
    e2u = file_mask.e2u.squeeze()

    # Masking the variables onto the Drake Passage

    u_masked = uo.isel(nav_lon=0) * umask_Drake
    e3u_masked = e3u.isel(nav_lon=0) * umask_Drake
    e2u_masked = e2u.isel(nav_lon=0) * umask_Drake

    # Multiplying zonal velocity by the sectional areas (e2u*e3u)

    ubar = u_masked * e3u_masked
    flux = (e2u_masked * ubar).sum(dim=["nav_lat", "depth"])

    # Returning Total Transport across Drake passage as a numpy scalar (unit : Sv)
    return flux / 1e6


def ACC_Drake_metric_2(uo, ssh, file_mask):
    """
    Metric Extraction Function :
    Antarctic Circumpolar Current Transport at the DINO equivalent of the Drake Passage (x=0).
    Unit : Sv


    Input :
       -    uo        : xarray.DataArray
       -    ssh       : xarray.DataArray
       -    file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files


    """

    e3u_0 = file_mask.e3u_0
    e2u = file_mask.e2u
    umask_Drake = file_mask.umask.isel(nav_lon=0)

    # Recomputing e3u, using ssh to refactor the original e3u_0 cell heights)

    ssh_u = (ssh + ssh.roll(_nav_lon=-1)) / 2
    bathy_u = e3u_0.sum(dim="depth")
    ssumask = umask_Drake[:, 0]
    e3u = e3u_0 * (1 + ssh_u * ssumask / (bathy_u + 1 - ssumask))

    # Masking the variables onto the Drake Passage

    u_masked = uo.isel(nav_lon=0) * umask_Drake
    e3u_masked = e3u.isel(nav_lon=0) * umask_Drake
    e2u_masked = e2u.isel(nav_lon=0) * umask_Drake

    # Multiplying zonal velocity by the sectional areas (e2u*e3u)

    ubar = (u_masked * e3u_masked).sum(dim="depth")
    flux = (e2u_masked * ubar).sum()

    # Returning Total Transport across Drake passage as a numpy scalar (unit : Sv)
    return flux.data / 1e6


def NASTG_BSF_max(vo, ssh, file_mask):
    """
    Metric Extraction Function :
    Intensity of the North-Atlantic SubTropical Gyre (NASTG) which contains the Gulf-Stream Current.
    Computed using the Barotropic Stream Function (BSF).
    Unit : Sv


    Input :
       -    vo        : xarray.DataArray
       -    file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files
    """

    e3v_0 = file_mask.e3v_0.squeeze()
    e1v = file_mask.e1v.squeeze()
    vmask = file_mask.vmask.squeeze()
    # Updating e3v from e3v_0 and SSH
    ssh_v = (ssh + ssh.roll(nav_lat=-1)) / 2
    bathy_v = e3v_0.sum(dim="depth")
    ssvmask = vmask.isel(depth=0)
    e3v = e3v_0 * (1 + ssh_v * ssvmask / (bathy_v + 1 - ssvmask))

    # Integrating Meridional Transport (e3v*e1v*vo) along depth and X from Western boundary eastward
    # to get Barotropic Stream Function with the "American continent" as reference point (BSF=0)
    V = (vo * e3v).sum(dim="depth")  #  == "Barotropic Velocity" * Bathymetry
    BSF = (V * e1v * ssvmask).cumsum(
        dim="nav_lon"
    ) / 1e6  # Integrating from the West, and converting from m³/s to Sv
    # Selecting 0N-40N window where to search for the maximum, which will correspond to the center of rotation for the gyre
    BSF_NASPG = BSF.where(abs(BSF.nav_lat - 20) < 20)

    return BSF_NASPG.max(dim=["nav_lat", "nav_lon"])
