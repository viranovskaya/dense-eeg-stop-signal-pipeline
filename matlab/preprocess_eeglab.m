function preprocess_eeglab(vhdr_file, output_dir, bad_channels, components_to_remove)
% Reproducible EEGLAB equivalent of the recovered preprocessing protocol.
%
% Example:
%   preprocess_eeglab('/data/sub-001_task-stop.vhdr', ...
%                     '/results/sub-001', ...
%                     {'CHAN1','CHAN2'}, []);
%
% Requirements: EEGLAB with the BrainVision importer and DIPFIT electrode file.

if nargin < 3, bad_channels = {}; end
if nargin < 4, components_to_remove = []; end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

[input_dir, input_name, input_ext] = fileparts(vhdr_file);
if ~strcmpi(input_ext, '.vhdr')
    error('Input must be a BrainVision .vhdr file.');
end

EEG = pop_loadbv(input_dir, [input_name input_ext]);

% Assign non-EEG types before reference and ICA decisions.
for channel_index = 1:numel(EEG.chanlocs)
    if strcmpi(EEG.chanlocs(channel_index).labels, 'ECG')
        EEG.chanlocs(channel_index).type = 'ECG';
    elseif strcmpi(EEG.chanlocs(channel_index).labels, 'EOG')
        EEG.chanlocs(channel_index).type = 'EOG';
    else
        EEG.chanlocs(channel_index).type = 'EEG';
    end
end

EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);

lookup_file = which('standard_1005.elc');
if isempty(lookup_file)
    error('standard_1005.elc was not found on the MATLAB/EEGLAB path.');
end
EEG = pop_chanedit(EEG, 'lookup', lookup_file);

% ECG is not needed further; EOG is retained as an auxiliary channel.
if any(strcmpi({EEG.chanlocs.labels}, 'ECG'))
    EEG = pop_select(EEG, 'rmchannel', {'ECG'});
end

% Dead electrodes are removed before referencing, then restored by spherical
% interpolation so that all participants retain the same scalp montage.
original_chanlocs = EEG.chanlocs;
if ~isempty(bad_channels)
    EEG = pop_select(EEG, 'rmchannel', bad_channels);
end
eog_index = find(strcmpi({EEG.chanlocs.type}, 'EOG'));
EEG = pop_reref(EEG, [], 'exclude', eog_index);
if ~isempty(bad_channels)
    EEG = pop_interp(EEG, original_chanlocs, 'spherical');
    eog_index = find(strcmpi({EEG.chanlocs.type}, 'EOG'));
    EEG = pop_reref(EEG, [], 'exclude', eog_index);
end

continuous_file = [input_name '_1-40Hz_avgref.set'];
EEG = pop_saveset(EEG, 'filename', continuous_file, 'filepath', output_dir);

% Fit one ICA decomposition on continuous EEG channels only. Components are
% removed only when their explicit indices are provided and therefore logged.
eeg_indices = find(strcmpi({EEG.chanlocs.type}, 'EEG'));
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, ...
                 'chanind', eeg_indices, 'rndreset', 'yes', 'interrupt', 'on');
if ~isempty(components_to_remove)
    EEG = pop_subcomp(EEG, components_to_remove, 0);
end
EEG = pop_saveset(EEG, 'savemode', 'resave');

% Go epochs preserve the S6/S4 response outcome events.
EEG_go = pop_epoch(EEG, {'S  17', 'S  18'}, [-1.5 3.0], 'epochinfo', 'yes');
EEG_go = pop_saveset(EEG_go, 'filename', [input_name '_go.set'], 'filepath', output_dir);

% Stop epochs are centered on S19 and split by whether S5 occurred.
EEG_stop_all = pop_epoch(EEG, {'S  19'}, [-2.0 2.0], 'epochinfo', 'yes');
has_s5 = false(1, EEG_stop_all.trials);
for trial_index = 1:EEG_stop_all.trials
    types = EEG_stop_all.epoch(trial_index).eventtype;
    if ~iscell(types), types = {types}; end
    has_s5(trial_index) = any(cellfun(@(x) ischar(x) && strcmp(strtrim(x), 'S  5'), types));
end

EEG_good_stop = pop_select(EEG_stop_all, 'trial', find(~has_s5));
EEG_bad_stop = pop_select(EEG_stop_all, 'trial', find(has_s5));
EEG_good_stop = pop_saveset(EEG_good_stop, 'filename', [input_name '_good_stop.set'], 'filepath', output_dir);
EEG_bad_stop = pop_saveset(EEG_bad_stop, 'filename', [input_name '_bad_stop.set'], 'filepath', output_dir);

fprintf('Saved continuous, go, good-stop, and bad-stop datasets to %s\n', output_dir);
end
