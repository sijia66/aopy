function convert_experiment_file_to_json(experiment_file_path,varargin)

% read experiment file saved at experiment_file_path, then reencode and
% save as a JSON file. If no json_file_path value is given, the JSON file
% is saved at the same location as experiment_file_path with the same name
% (up to the file extension) as experiment_file_path.

% Michael Nolan
% 2020.06.12

%% parse inputs
ip = inputParser;
default_json_file_path = strcat(strtok(experiment_file_path,'.'),'.json');
ip.addRequired('experiment_file_path',@(s)exist(s,'file'));
ip.addOptional('json_file_path',default_json_file_path,@(s)exist(fileparts(s),'dir'));
ip.parse(experiment_file_path,varargin{:});
experiment_file_path = ip.Results.experiment_file_path;
json_file_path = ip.Results.json_file_path;

%% write experiment data to output file
load(experiment_file_path,'experiment');
json_file_h = fopen(json_file_path,'w');
fwrite(json_file_h,jsonencode(experiment));
fclose(json_file_h);
