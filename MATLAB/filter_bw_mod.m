% =========================================================
% Чёрно-белый фильтр с модификацией (adaptive unsharp)
% =========================================================
clear; clc;

IMAGE_PATH = 'test_noisy.jpg';
OUT_PATH   = 'filtered_bw_mod.jpg';

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

img = imread(IMAGE_PATH);
img_gray = rgb2gray(img);
img_f = single(img_gray)/255.0;

UNSHARP_AMOUNT = 1.0;
MASK_RADIUS = 3;

% --- адаптивная карта градиентов ---
gx = conv2_rep(img_f, [1 0 -1; 2 0 -2; 1 0 -1]/8);
gy = conv2_rep(img_f, [1 2 1; 0 0 0; -1 -2 -1]/8);
grad_mag = sqrt(gx.^2 + gy.^2);
edge_mask = grad_mag; edge_mask = edge_mask-min(edge_mask(:));
edge_mask = edge_mask/max(edge_mask(:));
edge_mask = max(edge_mask,0.1);

% --- Gaussian blur для unsharp ---
[X,Y] = meshgrid(-MASK_RADIUS:MASK_RADIUS);
sigma = MASK_RADIUS/2;
G = exp(-(X.^2+Y.^2)/(2*sigma^2));
G = G/sum(G(:));
blurred = conv2_rep(img_f,G);

detail = img_f - blurred;
sharp = img_f + UNSHARP_AMOUNT * detail .* edge_mask;

sharp = max(min(sharp,1.0),0.0);
imwrite(uint8(sharp*255), OUT_PATH);
disp(['✅ Сохранено: ', OUT_PATH]);