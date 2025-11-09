function sharp_filter_nopkg2()
    % === Параметры ===
    a1 = 20;
    a2 = 2000;
    gain = 1;    % регулировка силы резкости
    image_path = 'test.jpg';

    % === Загрузка ===
    if ~isfile(image_path)
        error('Изображение не найдено.');
    end

    img = imread(image_path);

    % Перевод в оттенки серого
    if ndims(img) == 3
        gray = 0.2989 * double(img(:,:,1)) + ...
               0.5870 * double(img(:,:,2)) + ...
               0.1140 * double(img(:,:,3));
    else
        gray = double(img);
    end

    % Маска
    mask_base = [
        -1  -4  -8  -10  -8  -4  -1;
        -4  -16 -32 -40 -32 -16 -4;
        -8  -32  17  82  17 -32 -8;
        -10 -40  82 224  82 -40 -10;
        -8  -32  17  82  17 -32 -8;
        -4  -16 -32 -40 -32 -16 -4;
        -1  -4  -8  -10  -8  -4  -1
    ];

    mask_3x3 = [
        1 2 1;
        2 4 2;
        1 2 1
    ];

    mask_add = zeros(7,7);
    mask_add(3:5,3:5) = mask_3x3 * a1;
    mask_add(4,4) = mask_add(4,4) + a2;

    mask_final = mask_base + mask_add;
    mask_final = mask_final - mean(mask_final(:));
    mask_final = mask_final * gain;

    % === Вычисляем отклик фильтра (контуры) ===
    edge_enh = conv2(gray, mask_final, 'same');

    % === Комбинируем с исходным изображением ===
    sharp = gray + edge_enh;

    % --- Нормализация ---
    sharp = sharp - min(sharp(:));
    sharp = sharp / max(sharp(:));
    sharp = uint8(sharp * 255);

    gray = uint8(255 * (gray - min(gray(:))) / (max(gray(:)) - min(gray(:))));

    % === Отображение ===
    figure('Name','Фильтрация (усиленные контуры)','NumberTitle','off');
    subplot(1,2,1);
    imshow(gray);
    title('До фильтра');

    subplot(1,2,2);
    imshow(sharp);
    title(sprintf('После фильтра (a1=%d, a2=%d, gain=%.2f)', a1, a2, gain));

    imwrite(sharp, 'filtered_result_nopkg2.jpg');
    fprintf('Результат сохранён: filtered_result_nopkg2.jpg\n');
end
