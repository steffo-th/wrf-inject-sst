import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import glob
import os

# === 1. Läs in SST-data med flexiblare dimensionshantering ===
sst_file = "oisst-avhrr-v02r01.20250512_preliminary.nc"
print(f"Läser SST från: {sst_file}")
ds_sst = xr.open_dataset(sst_file)

# Undersök och skriv ut struktur för felsökning
print("SST-filens dimensioner:")
print(ds_sst.dims)
print("SST-variabelns struktur:")
print(ds_sst['sst'].dims)
print(ds_sst['sst'].shape)

# Mer flexibel extrahering av SST-data
sst_var = ds_sst['sst']
scale_factor = sst_var.attrs.get('scale_factor', 1.0)
fill_value = sst_var.attrs.get('_FillValue', -999)

# Anpassa extraheringen baserat på dimensioner
if len(sst_var.dims) == 4:  # t, z, y, x
    sst = sst_var.isel(time=0, zlev=0).astype('float32') * scale_factor
elif len(sst_var.dims) == 3:  # t, y, x
    sst = sst_var.isel(time=0).astype('float32') * scale_factor
elif len(sst_var.dims) == 2:  # y, x
    sst = sst_var.astype('float32') * scale_factor
else:
    print(f"Oväntad dimensionsstruktur: {sst_var.dims}")
    exit(1)

# Hantera saknade värden
if fill_value != -999:  # Om vi faktiskt hittade ett fill_value
    sst = sst.where(sst_var != fill_value)

# Konvertera till Kelvin
if sst.max() < 200:  # Troligen i Celsius
    print("Konverterar från Celsius till Kelvin")
    sst = sst + 273.15

# Hämta lat/lon-koordinater
lon = ds_sst['lon'].values
lat = ds_sst['lat'].values

# === 2. Läs in första met_em-fil för att kontrollera lon-system och domän ===
sample_files = sorted(glob.glob("met_em.d01.*.nc"))
if not sample_files:
    print("Inga met_em.d01.*.nc-filer hittades!")
    exit(1)
    
sample_met = sample_files[0]
print(f"Använder {sample_met} som mall")

with xr.open_dataset(sample_met) as ds_sample:
    # Kontrollera variabelnamn
    print("Tillgängliga variabler i met_em:")
    print(list(ds_sample.variables))
    
    # Anpassa för olika variabelnamn
    if "XLONG_M" in ds_sample:
        wrf_lons = ds_sample["XLONG_M"].isel(Time=0).values
        wrf_lats = ds_sample["XLAT_M"].isel(Time=0).values
    elif "XLONG" in ds_sample:
        wrf_lons = ds_sample["XLONG"].isel(Time=0).values
        wrf_lats = ds_sample["XLAT"].isel(Time=0).values
    else:
        print("Kan inte hitta longitud/latitud i met_em-filen!")
        exit(1)
        
    wrf_lon_min = wrf_lons.min()
    wrf_lon_max = wrf_lons.max()
    wrf_lat_min = wrf_lats.min()
    wrf_lat_max = wrf_lats.max()

# === 3. Kolla om vi behöver konvertera SST-lon till -180...180 ===
print(f"WRF longitude range: {wrf_lon_min:.2f} to {wrf_lon_max:.2f}")
print(f"SST longitude range: {lon.min():.2f} to {lon.max():.2f}")

convert_lons = False
if wrf_lon_min < 0 and lon.max() > 180:
    convert_lons = True
    lon = lon.copy()  # Skapa en kopia för att undvika att ändra originalet
    lon[lon > 180] -= 360
    print("WRF använder -180 till 180, OISST-longitudes konverteras.")
else:
    print("WRF och OISST-longitudes verkar redan matcha.")

# === 4. Kontroll av geografisk överlappning ===
sst_lon_min = lon.min()
sst_lon_max = lon.max()
sst_lat_min = lat.min()
sst_lat_max = lat.max()

print(f"WRF lat/lon: {wrf_lat_min:.2f}–{wrf_lat_max:.2f}, {wrf_lon_min:.2f}–{wrf_lon_max:.2f}")
print(f"OISST lat/lon: {sst_lat_min:.2f}–{sst_lat_max:.2f}, {sst_lon_min:.2f}–{sst_lon_max:.2f}")

lon_overlap = not (wrf_lon_max < sst_lon_min or wrf_lon_min > sst_lon_max)
lat_overlap = not (wrf_lat_max < sst_lat_min or wrf_lat_min > sst_lat_max)

if not (lon_overlap and lat_overlap):
    print("\n[VARNING] OISST täcker inte helt WRF-domänen!")
    print("Det kan resultera i saknade värden. Fortsätter ändå...")

# === 5. Interpoleringspunkter ===
print("Förbereder interpolation...")
lon2d, lat2d = np.meshgrid(lon, lat)
points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
values = sst.values.ravel()

# Försök hantera NaN-värden i källdata
mask = ~np.isnan(values)
if not np.any(mask):
    print("Alla SST-värden är NaN! Kontrollera källdata.")
    exit(1)
elif not np.all(mask):
    print(f"Hittade {np.sum(~mask)} NaN-värden i SST-data, använder bara giltiga punkter för interpolation.")
    points = points[mask]
    values = values[mask]

# === 6. Loop över met_em-filer ===
for inpath in sorted(glob.glob("met_em.d01.*.nc")):
    outpath = inpath  # Skriv över originalfilen
    print(f"\nBearbetar: {inpath}")

    with xr.open_dataset(inpath) as ds_in:
        # Identifiera rätt variabelnamn
        if "XLAT_M" in ds_in:
            lats = ds_in["XLAT_M"].isel(Time=0).values
            lons = ds_in["XLONG_M"].isel(Time=0).values
        else:
            lats = ds_in["XLAT"].isel(Time=0).values
            lons = ds_in["XLONG"].isel(Time=0).values

        # Kontrollera gridstorlek för snabbare interpolation
        print(f"  WRF-grid: {lats.shape}")
        if lats.size > 1000000:  # Stort grid kan vara minnes- och tidskrävande
            print("  Stort grid, interpolation kan ta tid...")

        try:
            print("  Interpolerar med linear metod...")
            sst_interp = griddata(points, values, (lats, lons), method='linear')
            
            # Kontrollera interpolationsresultat
            nan_percent = np.sum(np.isnan(sst_interp)) / sst_interp.size * 100
            print(f"  {nan_percent:.1f}% av interpolerade punkter är NaN")
            
            if nan_percent > 0:
                print("  Fyller i saknade värden med närmaste granne...")
                nearest = griddata(points, values, (lats, lons), method='nearest')
                sst_interp = np.where(np.isnan(sst_interp), nearest, sst_interp)
        except Exception as e:
            print(f"  [FEL] Interpolation misslyckades: {e}")
            print("  Försöker med närmaste granne istället...")
            sst_interp = griddata(points, values, (lats, lons), method='nearest')

        # Hitta en lämplig mall för dimensioner - använd LANDMASK istället för XLAND
        template_var = "LANDMASK"
        if template_var not in ds_in:
            # Leta efter alternativ
            for alt_var in ["HGT_M", "SKINTEMP", "TT"]:
                if alt_var in ds_in:
                    template_var = alt_var
                    print(f"  Använder {template_var} som mall för dimensioner")
                    break
        
        if template_var not in ds_in:
            print("  [FEL] Kan inte hitta lämplig malldimension i filen")
            continue

        # Skapa DataArray och injicera i filen
        sst_da = xr.DataArray(
            np.expand_dims(sst_interp.astype('float32'), axis=0),
            dims=ds_in[template_var].dims,  # Använd dimensioner från mallen
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

        # Skapa kopia av datasettet och lägg till SST
        ds_out = ds_in.copy(deep=True)
        ds_out["SST"] = sst_da
        
        # Spara till temporär fil och byt sedan namn för att undvika korrupta filer
        temp_outpath = outpath + ".tmp"
        ds_out.to_netcdf(temp_outpath)
        print("  SST injicerad, sparar...")
        
        # Kontrollera att filen skapades korrekt
        if os.path.exists(temp_outpath) and os.path.getsize(temp_outpath) > 0:
            os.rename(temp_outpath, outpath)
            print(f"  -> Klar: {outpath}")
        else:
            print(f"  [FEL] Problem med att skapa utfil: {temp_outpath}")

print("\nKlart! SST injicerad i alla met_em-filer.")
print("Kom ihåg att sätta sst_update = 1 i namelist.input")
