% -----------------------------------------------------------
% Цветной фильтр 7x7 + встроенная 3x3 апертура
% Полностью совместим с GNU Octave без пакетов
% -----------------------------------------------------------

clear; clc;

function out = conv2_rep(img, kernel)
    % conv2 с replicate padding, без padarray

    [h, w] = size(img);
    [kh, kw] = size(kernel);

    ph = floor(kh/2);
    pw = floor(kw/2);

    % ---- ручное replicate padding ----
    padded = zeros(h + 2*ph, w + 2*pw);

    % центр
    padded(ph+1:ph+h, pw+1:pw+w) = img;

    % верхняя и нижняя полосы
    padded(1:ph, pw+1:pw+w)      = repmat(img(1,:), ph, 1);
    padded(ph+h+1:end, pw+1:pw+w) = repmat(img(end,:), ph, 1);

    % левая и правая полосы
    padded(:, 1:pw)               = repmat(padded(:, pw+1), 1, pw);
    padded(:, pw+w+1:end)         = repmat(padded(:, pw+w), 1, pw);

    % углы
    padded(1:ph, 1:pw)                 = img(1,1);
    padded(1:ph, pw+w+1:end)           = img(1,end);
    padded(ph+h+1:end, 1:pw)           = img(end,1);
    padded(ph+h+1:end, pw+w+1:end)     = img(end,end);
    % ---------------------------------

    tmp = conv2(padded, kernel, "same");

    out = tmp(ph+1:ph+h, pw+1:pw+w);
endfunction


IMAGE_PATH = 'test_noisy.jpg';
OUT_PATH   = 'filtered_result_rgb_nomod_octave.jpg';

img = imread(IMAGE_PATH);
img_f = single(img) / 255.0;

a1 = 2;
a2 = 20;
gain = 0.25;

mask_base = single([
    -1 -4 -8 -10 -8 -4 -1;
    -4 -16 -32 -40 -32 -16 -4;
    -8 -32 17 82 17 -32 -8;
    -10 -40 82 224 82 -40 -10;
    -8 -32 17 82 17 -32 -8;
    -4 -16 -32 -40 -32 -16 -4;
    -1 -4 -8 -10 -8 -4 -1
]);

mask_3x3 = single([
    1 2 1;
    2 4 2;
    1 2 1
]);

mask_add = zeros(7,7,"single");
mask_add(3:5,3:5) = mask_3x3 .* a1;
mask_add(4,4) += a2;

mask_final = mask_base + mask_add;
mask_final = (mask_final - mean(mask_final(:))) * gain;

[h, w, ~] = size(img_f);
sharp = zeros(h, w, 3, "single");

for c = 1:3
    ch = img_f(:,:,c);
    filtered = conv2_rep(ch, mask_final);
    sharp(:,:,c) = ch + filtered;
end

sharp = max(min(sharp,1.0),0.0);

imwrite(uint8(sharp * 255), OUT_PATH);
disp(["Saved: ", OUT_PATH]);
