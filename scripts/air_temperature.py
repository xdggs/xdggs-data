import pathlib
import xarray as xr
import xdggs
from xdggs.healpix import HealpixInfo
from xdggs.h3 import H3Info

import numpy as np
import healpy as hp
import xvec
import shapely
import h3ronpy.arrow


def derive_bbox(ds, center_meridian=180):
    offset = 180 - center_meridian
    shifted = ds.assign_coords(lon=lambda ds: (ds["lon"] + offset) % 360 - offset)

    return shapely.box(
        shifted["lon"].min().item(),
        shifted["lat"].min().item(),
        shifted["lon"].max().item(),
        shifted["lat"].max().item(),
    )


def interpolate_to_healpix(ds, grid_info):
    bbox = derive_bbox(ds, center_meridian=0)

    subset = (
        xr.Dataset(
            coords={
                "cell_ids": (
                    "cells",
                    np.arange(12 * 4**grid_info.resolution),
                    grid_info.to_dict(),
                )
            }
        )
        .pipe(xdggs.decode)
        # select using the bbox
        # TODO: replace by polygon search
        .assign_coords(cell_boundaries=lambda ds: ds.dggs.cell_boundaries())
        .set_xindex("cell_boundaries")
        .xvec.set_geom_indexes("cell_boundaries")
        .sel({"cell_boundaries": bbox}, method="intersects")
        .drop_indexes(["cell_boundaries"])
        .drop_vars(["cell_boundaries"])
        .sortby("cell_ids")
    )

    grid = subset.pipe(lambda ds: ds.merge(ds.dggs.cell_centers()))
    return ds.interp(
        lon=grid["longitude"].variable, lat=grid["latitude"].variable, method="linear"
    ).assign_coords(cell_ids=grid["cell_ids"].variable)


def interpolate_to_h3(ds, grid_info):
    bbox = derive_bbox(ds, center_meridian=0)

    cell_ids = h3ronpy.arrow.vector.geometry_to_cells(
        bbox,
        resolution=grid_info.resolution,
        containment_mode=h3ronpy.ContainmentMode.ContainsBoundary,
    )
    grid = (
        xr.Dataset(coords={"cell_ids": ("cells", cell_ids, grid_info.to_dict())})
        .pipe(xdggs.decode)
        .pipe(lambda ds: ds.merge(ds.dggs.cell_centers()))
    )

    return ds.interp(
        lon=grid["longitude"].variable, lat=grid["latitude"].variable, method="linear"
    ).assign_coords(cell_ids=grid["cell_ids"].variable)


if __name__ == "__main__":
    ds = xr.tutorial.open_dataset("air_temperature").load()

    root = pathlib.Path.cwd() / "air_temperature"
    root.mkdir(exist_ok=True)

    healpix_info = HealpixInfo(resolution=4, indexing_scheme="nested")
    healpix_ds = interpolate_to_healpix(ds, healpix_info)
    healpix_ds.to_netcdf(root / "healpix.nc", mode="w")

    h3_info = H3Info(resolution=2)
    h3_ds = interpolate_to_h3(ds, h3_info)
    h3_ds.to_netcdf(root / "h3.nc", mode="w")
