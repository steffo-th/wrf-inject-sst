# wrf-inject-sst
Injects sea surface temperature (SST) from OISST NetCDF files into WRF's `met_em.d01.*.nc` files.

This tool reads daily SST data (e.g., from NOAA OISST v2.1) and injects it into WRF preprocessor output files before running real.exe. It handles interpolation to the WRF grid, missing values, and latitude/longitude system conversions.

## Features

- Supports multiple dimension formats in SST files
- Detects and handles `_FillValue` and missing data
- Converts SST to Kelvin if necessary
- Interpolates to WRF grid using `linear`, with fallback to `nearest`
- Overwrites the original `met_em` files (safe write via temporary file)
- No WRF rebuild required — just set `sst_update = 1` in `namelist.input`

## Requirements

- Python 3.7+
- `xarray`
- `numpy`
- `scipy`

Install with:

```bash
pip install xarray numpy scipy
