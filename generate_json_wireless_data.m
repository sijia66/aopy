% generate_json_wireless_data.m
%
% Use convert_experiment_file_to_json.m to save *experiment.mat files to
% *experiment.json files. This will facilitate python data analysis.

% Michael Nolan
% 2020.06.14

% data_dir_root = '/Volumes/G/Data/WirelessData/Goose_Multiscale_M1/';
date_dir_root = '/Volumes/GoogleDrive/Shared\ drives/aoLab/Data/WirelessData/Goose_Multiscale_M1/';
rec_date_list = dir(fullfile(data_dir_root,'18*'));
rec_date_list = {rec_date_list.name};
for rec_date = rec_date_list
    data_dir_date_root = fullfile(data_dir_root,rec_date{1});
    subdir_session_list = dir(fullfile(data_dir_date_root));
    subdir_session_list = {subdir_session_list.name};
    rec_session_idx = cellfun(@(x)~isnan(str2double(x)),subdir_session_list);
    rec_session_list = subdir_session_list(rec_session_idx);
    for rec_session = rec_session_list
        rec_session_dir = fullfile(data_dir_date_root,rec_session{1});
        experiment_file = dir(fullfile(rec_session_dir,'*experiment.mat'));
        if isempty(experiment_file)
            fprintf('no experiment file found in %s\n',rec_session_dir);
        else
            experiment_file = experiment_file(1).name;
            experiment_file_fullpath = fullfile(rec_session_dir,experiment_file);
            % create .json file from experiment file
            try
                convert_experiment_file_to_json(experiment_file_fullpath);
            catch err
                disp(err)
            end
        end
    end
end