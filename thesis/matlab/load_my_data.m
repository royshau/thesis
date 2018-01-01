function [real, imag, mask_all] = load_my_data(basic_path, tt, N, W, H, offset)
% Read my data
size_dat = W*H;
path_tt = fullfile(basic_path, tt, '000000.k_space_real_gt.bin');
f = fopen(path_tt, 'rb');
fseek(f, N*size_dat*offset, 0);
real = fread(f, N*size_dat, 'float32');
real = reshape(real, W, H, []);
fclose(f);

path_tt = fullfile(basic_path, tt, '000000.k_space_imag_gt.bin');
f = fopen(path_tt, 'rb');
fseek(f, N*size_dat*offset, 0);
imag = fread(f, N*size_dat, 'float32');
imag = reshape(imag, W, H, []);
fclose(f);

% 
% path_tt = fullfile(basic_path, tt, '000000.image_gt.bin');
% f = fopen(path_tt, 'rb');
% org_image = fread(f, N*size_dat, 'float32');
% org_image = reshape(org_image, W, H, []);
% fclose(f);


names = {'mask.bin','mask2_5.bin', 'mask4.bin', 'mask6.bin', 'mask8.bin', 'mask10.bin', 'mask20.bin'};
mask_all = cell(length(names),1);
index = 1;
for name = names
    path_tt = fullfile('/media/ohadsh/Data/ohadsh/work/matlab/thesis', name);
    f = fopen(path_tt{1}, 'rb');
    mask = fread(f, 1*size_dat, 'uint8');
    mask = reshape(mask, W, H, [])';
    fclose(f);
    mask_all{index} = mask;
    index = index + 1;
end

end

