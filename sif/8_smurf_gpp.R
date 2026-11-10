# 10 m SIF -> 10 m GPP with SMUrF. Needs the 10m branch of SMUrF
# (https://github.com/aiqiz/SMUrF/tree/10m) cloned locally, with its path in
# config.yaml (smurf.smurf_dir). Run from the conda env: SMUrF calls python3.
# This script only fills in the SMUrF settings from config.yaml and runs its run_gpp.r.
#
# in:  7.SIF_shoreline_corrected/<year>/SIF10m_8day_YYYYMMDD.nc
# out: 8.GPP_10m/easternCONUS/daily_mean_SIF_GPP_uncert_easternCONUS_<year>_10m.nc
#      8.GPP_10m/easternCONUS/daily/<year>/GPP10m_YYYYMMDD.nc

library(yaml)

file_arg <- grep('^--file=', commandArgs(), value = TRUE)
this_dir <- dirname(normalizePath(sub('^--file=', '', file_arg)))
cfg_file <- file.path(this_dir, '..', 'config.local.yaml')     # your own copy, not in git
if (!file.exists(cfg_file)) cfg_file <- file.path(this_dir, '..', 'config.yaml')
cfg      <- yaml::read_yaml(cfg_file)

# absolute paths as they are; relative ones inside data_dir, except data/... (shipped with the repo)
resolve <- function(p) {
    if (grepl('^/', p)) return(p)
    if (startsWith(p, 'data/')) return(normalizePath(file.path(this_dir, '..', p), mustWork = FALSE))
    file.path(cfg$data_dir, p)
}

region   <- cfg$sif$region
rcfg     <- cfg$sif$regions[[region]]
site_dir <- file.path(cfg$results_dir, 'sif', region)
smurf_wd_override <- normalizePath(cfg$smurf$smurf_dir)

# the variables run_gpp.r expects
all.yrs <- unlist(if (is.null(rcfg$years)) cfg$sif$years else rcfg$years)
minlon <- rcfg$smurf_bbox[[1]]; minlat <- rcfg$smurf_bbox[[2]]
maxlon <- rcfg$smurf_bbox[[3]]; maxlat <- rcfg$smurf_bbox[[4]]
sif.basepath <- file.path(site_dir, '7.SIF_shoreline_corrected')
lc.file      <- resolve(rcfg$smurf_landcover_file)
agb.path     <- resolve(cfg$smurf$agb_dir)
slp.file     <- file.path(smurf_wd_override, 'data', rcfg$smurf_slope_file)
c4.tag       <- rcfg$smurf_c4_tag
urban_veg_file <- if (is.null(rcfg$smurf_urban_veg_file) || rcfg$smurf_urban_veg_file == '') NULL else
                      file.path(smurf_wd_override, 'data', rcfg$smurf_urban_veg_file)
output.path  <- file.path(site_dir, '8.GPP_10m')
tmpdir       <- cfg$smurf$tmp_dir

source(file.path(smurf_wd_override, 'run_gpp.r'))
