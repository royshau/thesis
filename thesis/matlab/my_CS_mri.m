% This script demonstare a CS reconstruction from 
% Randomly undersmapled phase encodes of 2D FSE
% of a brain image.

% profile on
addpath(strcat(pwd,'/utils'));
addpath('/home/ohadsh/work/matlab/thesis/');

if exist('FWT2_PO') <2
	error('must have Wavelab installed and in the path');
end

% load brain512
% Load my data:
CS_ITERS = 1;
sampling_factor = 4;
start_line = 1;
keep_center = 0.05;
W = 256;
H = 256;
disp(['Working on radom mask - ', num2str(sampling_factor)])

size_dat = W * H;
basic_path = '/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/';
tt = 'train';

f_out = fopen('/sheard/googleDrive/Master/runs/CS/IXI/random_mask_Nov14/factor4/results.txt', 'w');
f_out_predict = fopen('/sheard/googleDrive/Master/runs/CS/IXI/random_mask_Nov14/factor4/cs_mri_predict.bin', 'wb');

N_imgs = 50;

[real_all, imag_all, mask_all] = get_data_memmap(basic_path, tt, W, H);
for i = 1:N_imgs

    %%

    disp(['Running on i=', num2str(i)]);
    
    real = real_all.Data(i).data;
    image = imag_all.Data(i).data;
    data = real + 1i*imag;
    
    org_image = abs(norm_ifft(data));
    
    %         imagesc(org_image); colormap gray;
%     %     pdf_org = pdf; mask_org = mask;
%     pdf = imresize(pdf, 0.5);


    % CS Mask
    % mask = imresize(mask, 0.5);

    % % My Mask
%     mask = get_mask(W, H, sampling_factor, start_line, keep_center);
    mask = get_random_mask(sampling_factor);

    data_used = sum(mask(:)) / length(mask(:));

%     pdf = ones('like', pdf);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % L1 Recon Parameters 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    N = size(data); 	% image Size
    DN = size(data); 	% data Size
    TVWeight = 0.002; 	% Weight for TV penalty
    xfmWeight = 0.005;	% Weight for Transform L1 penalty
    Itnlim = 8;		% Number of iterations


    %generate Fourier sampling operator
    FT = p2DFT(mask, N, 1, 2);

    % scale data
    % im_dc = FT'*(data.*mask./pdf);
    % data = data/max(abs(im_dc(:)));
    % im_dc = im_dc/max(abs(im_dc(:)));
    im_dc = abs(norm_ifft(data.*mask));

    % imagesc(abs(im_dc)); colormap('gray');
    % waitforbuttonpress;

    %generate transform operator
    XFM = Wavelet('Daubechies',4,4);	% Wavelet

    % initialize Parameters for reconstruction
    param = init;
    param.FT = FT;
    param.XFM = XFM;
    param.TV = TVOP;
    param.data = data;
    param.TVWeight =TVWeight;     % TV penalty 
    param.xfmWeight = xfmWeight;  % L1 wavelet penalty
    param.Itnlim = Itnlim;

%     figure(100), imshow(abs(im_dc),[]);drawnow;

    res = XFM*im_dc;

    % do iterations
    tic
    for n=1:CS_ITERS
        verbose = false;
        res = fnlCg(res,param, verbose);
        im_res = XFM'*res;
%         figure(100), imshow(abs(im_res),[]), drawnow
    end
    toc


%         figure, imshow(abs(cat(2,org_image, im_dc,im_res)),[]); colorbar;
%         figure, imshow(abs(cat(2,im_dc(50:150,60:200), im_res(50:150,60:200))),[0,1],'InitialMagnification',200);

%     title(' zf-w/dc              l_1 Wavelet');

    psnr_cs = calc_psnr(sum((abs(im_res(:)) - org_image(:)).^2));
    psnr_zero = calc_psnr(sum((abs(im_dc(:)) - org_image(:)).^2));

    disp('#############################')
    t1 = ['CS=', num2str(psnr_cs)];
    disp(t1);
    t2 = ['Zero=', num2str(psnr_zero)];
    disp(t2);
    disp('#############################')
    fprintf(f_out, t1);
    fprintf(f_out, '\n');
    fprintf(f_out, t2);
    fprintf(f_out, '\n');

    fwrite(f_out_predict, abs(im_res)', 'float32');
    close all;
end
fclose('all');

% profile off
% profile viewer