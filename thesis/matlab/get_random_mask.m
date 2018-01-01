function mask = get_random_mask(sampling_factor)
    %GET_MASK Summary of this function goes here
    %   Detailed explanation goes here
    % Read my data
    basic_path = '/media/ohadsh/Data/ohadsh/work/matlab/thesis/';
    W = 256;
    H = 256;
    size_dat = W*H;
    path_mask = fullfile(basic_path, ['mask', num2str(sampling_factor), '.bin']);
    f = fopen(path_mask, 'rb');
    mask = fread(f, 1*size_dat, 'uint8');
    mask = reshape(mask, W, H, []);
    fclose(f);
end

