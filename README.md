# sifcam-urban-gpp: urban GPP from SIF and cameras

Code for estimating urban gross primary productivity (GPP) in two independent
ways and comparing them:

- **SIF**: satellite solar-induced fluorescence (TROPOMI, GOSIF) downscaled
  to 10 m with Sentinel-2 NIRv, then converted to GPP with SMUrF.
- **Cameras**: greenness (GCC) from traffic cameras and phenocams, converted
  to GPP with the DPPM gatekeeper model.

The method is described in the paper: Zhang et al. (in preparation), citation
to be added. Data for the Toronto case are on Zenodo: DOI to be added.

## Setup

```
conda env create -f environment.yml
conda activate urban-gpp
```

Then copy `config.yaml` to `config.local.yaml` and put your paths in it
(`data_dir`, `results_dir`, the SMUrF and DPPM folders, and the raw data
folders under `data_dir`). Every script reads `config.local.yaml` if it
exists, otherwise `config.yaml`; the local copy is ignored by git.

Two outside pieces are needed:
- **SMUrF, 10m branch**: https://github.com/aiqiz/SMUrF/tree/10m. Clone it,
  then set `smurf.smurf_dir`. The R packages it needs are in `environment.yml`.
- **DPPM** (Restrepo-Coupe et al., "Phenology and the City"): the model code
  is not part of this repository; ask the authors for
  `FN_computeGEPgatekeeperGCCgn2br.m` and set `camera.dppm_dir` to its folder.
  The fitted parameters (Table 1 of that paper) are in
  `camera/dppm_parameters.csv`.

Downloading Sentinel-2 needs a free Copernicus Data Space account. Put your
client id and secret in the environment variables `SH_CLIENT_ID` and
`SH_CLIENT_SECRET`.

## Run

Run the scripts from inside their folder, in this order, with the conda env
active (also for `Rscript`, SMUrF calls Python for the daily files):

```
cd sif
python 1_s2_download.py
python 2_nirv_composite.py
python 3_nirv_gapfill.py
python 4_tropomi_downscale.py
python 5_gosif_downscale.py
python 6_merge_gapfill.py
python 7_shoreline.py
Rscript 8_smurf_gpp.R

cd ../camera
python 1_traffic_gcc.py       # traffic cameras
python 1_phenocam_align.py    # phenocams
python 2_phenocam_gcc.py
python 3_dppm_input.py
matlab -batch "dppm_run"        # step 4 (MATLAB script names cannot start with a digit)

cd ../comparison
python 1_extract_at_sites.py
python 2_compare.py
```

Each script starts with a short note on what it reads and writes. All
outputs go under `results_dir`. A step that has finished a year leaves an
empty `_complete` file in its output folder and is skipped next time; delete
that file to redo the year.

The SIF scripts process one region at a time (`sif.region` in
`config.yaml`). For the flux tower comparison, run them once for `toronto`
and once for `turkey_point`. The Turkey Point results in the paper started
from NIRv mosaics made earlier with the same method (steps 0 and 1 of the
output folder), not from a fresh Sentinel-2 download.

`data/` holds the small inputs that are not downloaded: the camera to
weather station mapping, daily temperature and PAR, and the phenocam
reference frames with their labelme vegetation polygons.
