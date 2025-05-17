import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import glob
import os

# === 1. Load SST data with flexible dimension handling ===
sst_file = "oisst-avhrr-v02r01.20250512_preliminary.nc"
print(f"Reading SST from: {sst_file}")
ds_sst = xr.open_dataset(sst_file)

# Examine file structure for debugging
print("SST file dimensions:")
print(ds_sst.dims)
print("SST variable dimensions:")
print(ds_sst['sst'].dims)
print(ds_sst['sst'].shape)

# More flexible SST extraction
sst_var = ds_sst['sst']
scale_factor = sst_var.attrs.get('scale_factor', 1.0)
fill_value = sst_var.attrs.get('_FillValue', -999)

# Extract SST depending on dimension structure
if len(sst_var.dims) == 4:  # time, zlev, y, x
    sst = sst_var.isel(time=0, zlev=0).astype('float32') * scale_factor
elif len(sst_var.dims) == 3:  # time, y, x
    sst = sst_var.isel(time=0).astype('float32') * scale_factor
elif len(sst_var.dims) == 2:  # y, x
    sst = sst_var.astype('float32') * scale_factor
else:
    print(f"Unexpected dimension structure: {sst_var.dims}")
    exit(1)

# Handle missing values
if fill_value != -999:
    sst = sst.where(sst_var != fill_value)

# Convert to Kelvin if needed
if sst.max() < 200:
    print("Converting from Celsius to Kelvin")
    sst = sst + 273.15

# Get lat/lon coordinates
lon = ds_sst['lon'].values
lat = ds_sst['lat'].values

# === 2. Load a sample met_em file to check domain and longitude system ===
sample_files = sorted(glob.glob("met_em.d01.*.nc"))
if not sample_files:
    print("No met_em.d01.*.nc files found!")
    exit(1)

sample_met = sample_files[0]
print(f"Using {sample_met} as template")

with xr.open_dataset(sample_met) as ds_sample:
    # Identify coordinate variable names
    print("Variables in met_em:")
    print(list(ds_sample.variables))

    if "XLONG_M" in ds_sample:
        wrf_lons = ds_sample["XLONG_M"].isel(Time=0).values
        wrf_lats = ds_sample["XLAT_M"].isel(Time=0).values
    elif "XLONG" in ds_sample:
        wrf_lons = ds_sample["XLONG"].isel(Time=0).values
        wrf_lats = ds_sample["XLAT"].isel(Time=0).values
    else:
        print("Could not find longitude/latitude in met_em file!")
        exit(1)

    wrf_lon_min = wrf_lons.min()
    wrf_lon_max = wrf_lons.max()
    wrf_lat_min = wrf_lats.min()
    wrf_lat_max = wrf_lats.max()

# === 3. Convert SST longitudes to match WRF if needed ===
print(f"WRF longitude range: {wrf_lon_min:.2f} to {wrf_lon_max:.2f}")
print(f"SST longitude range: {lon.min():.2f} to {lon.max():.2f}")

convert_lons = False
if wrf_lon_min < 0 and lon.max() > 180:
    convert_lons = True
    lon = lon.copy()
    lon[lon > 180] -= 360
    print("WRF uses -180 to 180, converting OISST longitudes.")
else:
    print("WRF and OISST longitude systems seem to match.")

# === 4. Check geographical overlap ===
sst_lon_min = lon.min()
sst_lon_max = lon.max()
sst_lat_min = lat.min()
sst_lat_max = lat.max()

print(f"WRF lat/lon: {wrf_lat_min:.2f}–{wrf_lat_max:.2f}, {wrf_lon_min:.2f}–{wrf_lon_max:.2f}")
print(f"OISST lat/lon: {sst_lat_min:.2f}–{sst_lat_max:.2f}, {sst_lon_min:.2f}–{sst_lon_max:.2f}")

lon_overlap = not (wrf_lon_max < sst_lon_min or wrf_lon_min > sst_lon_max)
lat_overlap = not (wrf_lat_max < sst_lat_min or wrf_lat_min > sst_lat_max)

if not (lon_overlap and lat_overlap):
    print("\n[WARNING] OISST does not fully cover the WRF domain!")
    print("This may result in missing values. Continuing anyway...")

# === 5. Prepare interpolation points ===
print("Preparing interpolation...")
lon2d, lat2d = np.meshgrid(lon, lat)
points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
values = sst.values.ravel()

# Filter out NaNs
mask = ~np.isnan(values)
if not np.any(mask):
    print("All SST values are NaN! Check your input.")
    exit(1)
elif not np.all(mask):
    print(f"Found {np.sum(~mask)} NaN values in SST, using only valid points.")
    points = points[mask]
    values = values[mask]

# === 6. Loop through all met_em files ===
for inpath in sorted(glob.glob("met_em.d01.*.nc")):
    outpath = inpath  # Overwrite original file
    print(f"\nProcessing: {inpath}")

    with xr.open_dataset(inpath) as ds_in:
        if "XLAT_M" in ds_in:
            lats = ds_in["XLAT_M"].isel(Time=0).values
            lons = ds_in["XLONG_M"].isel(Time=0).values
        else:
            lats = ds_in["XLAT"].isel(Time=0).values
            lons = ds_in["XLONG"].isel(Time=0).values

        print(f"  WRF grid size: {lats.shape}")
        if lats.size > 1_000_000:
            print("  Large grid detected, interpolation may take time...")

        try:
            print("  Interpolating using linear method...")
            sst_interp = griddata(points, values, (lats, lons), method='linear')

            nan_percent = np.sum(np.isnan(sst_interp)) / sst_interp.size * 100
            print(f"  {nan_percent:.1f}% of interpolated points are NaN")

            if nan_percent > 0:
                print("  Filling missing values using nearest neighbor...")
                nearest = griddata(points, values, (lats, lons), method='nearest')
                sst_interp = np.where(np.isnan(sst_interp), nearest, sst_interp)
        except Exception as e:
            print(f"  [ERROR] Linear interpolation failed: {e}")
            print("  Falling back to nearest neighbor interpolation...")
            sst_interp = griddata(points, values, (lats, lons), method='nearest')

        # Choose a reference variable to copy dimensions from
        template_var = "LANDMASK"
        if template_var not in ds_in:
            for alt_var in ["HGT_M", "SKINTEMP", "TT"]:
                if alt_var in ds_in:
                    template_var = alt_var
                    print(f"  Using {template_var} as dimension template")
                    break

        if template_var not in ds_in:
            print("  [ERROR] No suitable template variable found in file")
            continue

        # Create new DataArray for SST
        sst_da = xr.DataArray(
            np.expand_dims(sst_interp.astype('float32'), axis=0),
            dims=ds_in[template_var].dims,
            coords=ds_in[template_var].coords,
            attrs={
                "FieldType": 104,
                "MemoryOrder": "XY",
                "units": "K",
                "description": "Sea Surface Temperature",
                "stagger": "M",
                "sr_x": 1,
                "sr_y": 1
            }
        )

        # Create a copy and inject SST
        ds_out = ds_in.copy(deep=True)
        ds_out["SST"] = sst_da

        # Save to temp file first to avoid corruption
        temp_outpath = outpath + ".tmp"
        ds_out.to_netcdf(temp_outpath)
        print("  SST injected, saving...")

        if os.path.exists(temp_outpath) and os.path.getsize(temp_outpath) > 0:
            os.rename(temp_outpath, outpath)
            print(f"  -> Done: {outpath}")
        else:
            print(f"  [ERROR] Failed to create output file: {temp_outpath}")

print("\nDone! SST has been injected into all met_em files.")
print("No need to set sst_update = 1 in namelist.input")
