fn='mypictures/qh_wy_s3_c1_gr.png';
img = imread(fn);
h = fspecial('gaussian', [5, 5], 1.0);
img2 = imfilter(img, h, 'replicate');

imwrite(img2, 'mypictures/s3_fl.png')