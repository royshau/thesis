run('/media/ohadsh/Data/ohadsh/work/matlab/wavelet/SetPath4Wavelets.m')
addpath('/media/ohadsh/Data/ohadsh/work/matlab/sparseMRI_v0.2/utils/');
addpath('/media/ohadsh/Data/ohadsh/work/matlab/thesis/');

if exist('FWT2_PO') <2
	error('must have Wavelab installed and in the path');
end
% Load my data:
sampling_factor = 2;
start_line = 1;
keep_center = 0.05;
W = 256;
H = 256;
N = 2;
size_dat = W * H;
basic_path = '/sheard/Ohad/thesis/data/IXI/data_for_train/T1/sagittal/shuffle/';
tt = 'train';
[real_all, imag_all, mask_all] = load_my_data(basic_path, tt, N, W, H);

mask_id = 6;
mask = mask_all{mask_id}';
i = 1;
real = squeeze(real_all(:, :, i));
imag = squeeze(imag_all(:, :, i));
data = real + 1i*imag;

org_image = abs(norm_ifft(data));

% % My Mask
% mask = get_mask(W, H, sampling_factor, start_line, keep_center);
data_used = sum(mask(:)) / length(mask(:));

%% parameter setting
orig_ft     = fft_coil(data);
orig_ssos   = norm_ifft(data);

m=5; % filter size (shoud be odd)
n=5; % filter size (shoud be odd)

ddata           = data;
tol_set         = [1e-1 1e-2 1e-3 1e-4];
muiter_set      = [10 20 30 40];
sroi            = 2.0;
param           = struct('dname','brain_SE','data',ddata,'mask',mask,...
    'm', m, 'n', n,'mu',1e3,'muiter_set',muiter_set,'tol_set',tol_set,'sroi',sroi);

%% kxky ALOHA 
rec         = aloha_kxky(param);
s_rec       = norm_ifft(rec.recon);
s_rec_zero  = abs(norm_ifft(ddata.*mask));
figure,colormap gray;
subplot(1,3,1); imagesc(orig_ssos); title('Original');
subplot(1,3,2); imagesc(abs(s_rec_zero)); title('Zero');
subplot(1,3,3); imagesc(abs(s_rec)); title('ALOHA');

psnr_cs = calc_psnr(sum((abs(s_rec(:)) - orig_ssos(:)).^2))
psnr_zero = calc_psnr(sum((abs(s_rec_zero(:)) - orig_ssos(:)).^2))