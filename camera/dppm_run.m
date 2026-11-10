% Daily GCC + temperature + PAR -> GPP with the DPPM gatekeeper model
% (Gn2Br version, FN_computeGEPgatekeeperGCCgn2br). The DPPM code itself is not
% part of this repository; set camera.dppm_dir in config.yaml to the folder
% that has FN_computeGEPgatekeeperGCCgn2br.m.
%
% Parameters: dppm_parameters.csv = Table 1 of the DPPM paper (Restrepo-Coupe
% et al., "Phenology and the City"): Deciduous, Conifer (used as Evergreen) and
% the US-INa turfgrass fit (used as Grasses), with their 95 % CIs.
% We don't know which plant type a camera ROI sees, so each GCC stream is run
% with every parameter set that fits it (tree -> Deciduous + Evergreen,
% grass -> Grasses, phenocam ROI -> all three). Uncertainty: 500 Monte Carlo
% draws from the parameter CI95 (independent normal).
%
% in:  camera/dppm_input/   (from 3_dppm_input.py)
% out: camera/gpp/traffic/<year>/locXXXX.csv, camera/gpp/phenocam/<camera>.csv
%      columns GPP_<stream>_<PFT>, _MCmean, _p025, _p975   [gC m-2 d-1]

clear; clc;
here = fileparts(mfilename('fullpath'));
cfg_file = fullfile(here, '..', 'config.local.yaml');      % your own copy, not in git
if ~isfile(cfg_file), cfg_file = fullfile(here, '..', 'config.yaml'); end
cfg = fileread(cfg_file);
results_dir = strtrim(regexp(cfg, '^results_dir:\s*([^#\n]+)', 'tokens', 'once', 'lineanchors'));
S = jsondecode(fileread(fullfile(results_dir{1}, 'camera', 'dppm_input', 'settings.json')));
addpath(S.dppm_dir);

pfts = ["Deciduous", "Evergreen", "Grasses"];
stream_pfts = struct('tree', {{"Deciduous", "Evergreen"}}, ...
                     'grass', {{"Grasses"}}, ...
                     'gcc', {{"Deciduous", "Evergreen", "Grasses"}});
for p = pfts
    [P.(p).mean, P.(p).ci95] = read_params(S.params_csv, p);
end

% traffic cameras, one year at a time
for y = S.years(:)'
    in_dir  = fullfile(S.input_dir, 'traffic', string(y));
    out_dir = fullfile(S.output_dir, 'traffic', string(y));
    run_folder(in_dir, out_dir, P, stream_pfts, S.n_mc);
end
% phenocams, all years in one series
run_folder(fullfile(S.input_dir, 'phenocam'), fullfile(S.output_dir, 'phenocam'), P, stream_pfts, S.n_mc);


function run_folder(in_dir, out_dir, P, stream_pfts, n_mc)
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    files = dir(fullfile(in_dir, '*.csv'));
    files = files(~startsWith({files.name}, '.'));
    for i = 1:numel(files)
        T = readtable(fullfile(files(i).folder, files(i).name), 'VariableNamingRule', 'preserve');
        T.date = datetime(T.date);
        % Day number such that datetime(1,1,day) == date, so the model's
        % per-year GCC normalization follows calendar years even with gaps.
        day = days(T.date - datetime(1, 1, 0));

        streams = string(T.Properties.VariableNames(startsWith(T.Properties.VariableNames, 'GCC')));
        rng(42);   % same draws for every camera
        for s = streams
            name = lower(erase(s, ["GCC_", "GCC"]));
            if name == ""; name = "gcc"; end
            gcc = T.(s);
            for pft = string(stream_pfts.(char(name)))
                pm = P.(pft).mean; sd = P.(pft).ci95 / 1.96;
                g0 = FN_computeGEPgatekeeperGCCgn2br(pm, day, T.PPFD_umol, T.TA_C, gcc, 1);

                draws = pm + sd .* randn(n_mc, numel(pm));
                draws(:, [1 2 4 6 8]) = max(draws(:, [1 2 4 6 8]), 1e-6);   % positive params
                draws(:, [3 5]) = min(max(draws(:, [3 5]), 0), 1);          % x0 in [0, 1]
                ens = nan(height(T), n_mc);
                for j = 1:n_mc
                    ens(:, j) = clean(FN_computeGEPgatekeeperGCCgn2br(draws(j, :), day, T.PPFD_umol, T.TA_C, gcc, 1));
                end
                tag = "GPP_" + name + "_" + pft;
                T.(tag) = clean(g0);
                T.(tag + "_MCmean") = mean(ens, 2, 'omitnan');
                T.(tag + "_p025") = prctile(ens, 2.5, 2);
                T.(tag + "_p975") = prctile(ens, 97.5, 2);
            end
        end
        writetable(T, fullfile(out_dir, files(i).name));
        fprintf('%s %s\n', out_dir, files(i).name);
    end
end

function [m, c] = read_params(csv_path, pft)
    T = readtable(csv_path, 'VariableNamingRule', 'preserve');
    names = ["GEP_max", "k_green-up", "x0_green-up", "k_brown-down", ...
             "x0_brown-down", "PAR_half", "T_opt", "T_amp"];
    cols = struct('Deciduous', ["Deciduous", "Deciduous_CI95"], ...
                  'Evergreen', ["Evergreen_Mean", "Evergreen_CI95"], ...
                  'Grasses', ["Grasses_Mean", "Grasses_CI95"]);
    cc = cols.(char(pft));
    m = nan(1, numel(names)); c = m;
    for k = 1:numel(names)
        idx = strcmp(strtrim(string(T.Parameter)), names(k));
        m(k) = T.(cc(1))(idx);
        c(k) = T.(cc(2))(idx);
    end
end

function x = clean(x)
    x = double(real(x(:)));
    x(~isfinite(x)) = NaN;
end
