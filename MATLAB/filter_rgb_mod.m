clear; clc;

IMAGE_PATH = 'test_noisy.jpg';
OUT_PATH   = 'filtered_rgb_mod_octave_fast.jpg';

UNSHARP_AMOUNT = 1.8; % сила усиления контуров
MASK_RADIUS = 3;
GAUSS_SIGMA = 1.5;    % Gaussian для шумоподавления

% ---------------------------------------------------------
% conv2 с replicate padding
% ---------------------------------------------------------
function out = conv2_rep(img, kernel)
    [h, w] = size(img);
    [kh, kw] = size(kernel);
    ph = floor(kh/2); pw = floor(kw/2);

    padded = zeros(h+2*ph, w+2*pw);
    padded(ph+1:ph+h, pw+1:pw+w) = img;

    padded(1:ph, pw+1:pw+w) = repmat(img(1,:), ph,1);
    padded(ph+h+1:end, pw+1:pw+w) = repmat(img(end,:), ph,1);
    padded(:,1:pw) = repmat(padded(:,pw+1),1,pw);
    padded(:,pw+w+1:end) = repmat(padded(:,pw+w),1,pw);

    padded(1:ph,1:pw) = img(1,1); padded(1:ph,pw+w+1:end) = img(1,end);
    padded(ph+h+1:end,1:pw) = img(end,1); padded(ph+h+1:end,pw+w+1:end) = img(end,end);

    tmp = conv2(padded, kernel, 'same');
    out = tmp(ph+1:ph+h, pw+1:pw+w);
endfunction

% ---------------------------------------------------------
% Быстрый медианный фильтр 3x3 без вложенных циклов
% ---------------------------------------------------------
function out = med3x3_fast(img)
    [h,w] = size(img);
    % replicate padding вручную
    padded = zeros(h+2, w+2);
    padded(2:1+h, 2:1+w) = img;
    padded(1,:) = padded(2,:); padded(end,:) = padded(end-1,:);
    padded(:,1) = padded(:,2); padded(:,end) = padded(:,end-1,:);
    
    % формируем все 3x3 блоки в столбцы
    cols = zeros(9, h*w);
    idx = 1;
    for i = 1:3
        for j = 1:3
            block = padded(i:i+h-1, j:j+w-1);
            cols(idx,:) = block(:)';
            idx = idx + 1;
        end
    end
    out = median(cols, 1);
    out = reshape(out, h, w);
endfunction

% ---------------------------------------------------------
% Загрузка изображения
% ---------------------------------------------------------
img = imread(IMAGE_PATH);
img_f = single(img)/255.0; % [0,1]
[h, w, ~] = size(img_f);

% ---------------------------------------------------------
% RGB -> YCbCr вручную
% ---------------------------------------------------------
R = img_f(:,:,1); G = img_f(:,:,2); B = img_f(:,:,3);
Y  = 0.299*R + 0.587*G + 0.114*B;
Cb = 0.5 - 0.168736*R - 0.331264*G + 0.5*B;
Cr = 0.5 + 0.5*R - 0.418688*G - 0.081312*B;

% ---------------------------------------------------------
% Сглаживание Y
% ---------------------------------------------------------
hsize = 5;
[X,Yg] = meshgrid(-floor(hsize/2):floor(hsize/2));
Gk = exp(-(X.^2+Yg.^2)/(2*GAUSS_SIGMA^2));
Gk = Gk/sum(Gk(:));

Y_smooth = conv2_rep(Y, Gk);

% быстрый медианный фильтр
Y_med = med3x3_fast(Y_smooth);
Y_filt = Y_med;

% ---------------------------------------------------------
% Adaptive unsharp на Y
% ---------------------------------------------------------
gx = conv2_rep(Y_filt, [1 0 -1;2 0 -2;1 0 -1]/8);
gy = conv2_rep(Y_filt, [1 2 1;0 0 0;-1 -2 -1]/8);
grad_mag = sqrt(gx.^2 + gy.^2);
edge_mask = grad_mag; edge_mask = edge_mask - min(edge_mask(:));
edge_mask = edge_mask / max(edge_mask(:));
edge_mask = max(edge_mask, 0.1);

[Xb,Yb] = meshgrid(-MASK_RADIUS:MASK_RADIUS);
sigma = MASK_RADIUS/2;
G_blur = exp(-(Xb.^2 + Yb.^2)/(2*sigma^2));
G_blur = G_blur / sum(G_blur(:));
blurred = conv2_rep(Y_filt, G_blur);

detail = Y_filt - blurred;
Y_sharp = Y_filt + UNSHARP_AMOUNT * detail .* edge_mask;
Y_sharp = max(min(Y_sharp,1.0),0.0);

% ---------------------------------------------------------
% Cb/Cr Gaussian
% ---------------------------------------------------------
Cb_filt = conv2_rep(Cb, Gk);
Cr_filt = conv2_rep(Cr, Gk);

% ---------------------------------------------------------
% YCbCr -> RGB
% ---------------------------------------------------------
R_out = Y_sharp + 1.402*(Cr_filt-0.5);
G_out = Y_sharp - 0.344136*(Cb_filt-0.5) - 0.714136*(Cr_filt-0.5);
B_out = Y_sharp + 1.772*(Cb_filt-0.5);

sharp_rgb = cat(3,R_out,G_out,B_out);
sharp_rgb = max(min(sharp_rgb,1.0),0.0);

imwrite(uint8(sharp_rgb*255), OUT_PATH);
disp(['✅ Сохранено: ', OUT_PATH]);
