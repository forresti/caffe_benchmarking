
function alexnet_plus_one_desc()
    %img = single(zeros(227, 227, 3));
    %img(100, 100, :) = 255.0; %one white pixel
    %img(200, 200, :) = 255.0; %one white pixel
    img = imread('/media/big_disk/installers_old/caffe/voc-release5/cachedir/VOC2007/JPEGImages/000023.jpg');
    %figure(1)
    %imshow(img);
    %title('original image')

    model_file = '../../examples/alexnet_train_iter_470000';

    %243x243 test
    model_def_file = '/media/big_disk/installers_old/caffe/examples/imagenet_deploy_batchsize1_input_243x243_output_conv5.prototxt'; 
    caffe('init', model_def_file, model_file);
    caffe('set_mode_gpu');
    caffe('set_phase_test');
    img_243 = img(1:243, 1:243, :);
    desc_14 = AlexNet_descriptor(img_243);
    desc_14 = reshape(desc_14{1}, [14 14 256]); %243x243 -> 14x14 
    show_desc(desc_14, 4, '243x243 -> 14x14, conv5');

    %227x227 test
    model_def_file = '/media/big_disk/installers_old/caffe/examples/imagenet_deploy_batchsize1_output_conv5.prototxt';
    caffe('init', model_def_file, model_file);
    caffe('set_mode_gpu');
    caffe('set_phase_test');
    img_227 = img(1:227, 1:227, :);
    desc_13 = AlexNet_descriptor(img_227);
    desc_13 = reshape(desc_13{1}, [13 13 256]); %227x227 -> 13x13    
    show_desc(desc_13, 2, '227x227 -> 13x13, conv5');


    display('distance between desc_13 and top-left of desc_14:')
    dist = distance_metric(desc_13, desc_14(1:13, 1:13, :)) 

    display('distance between desc_13 and all zeros:')
    dist = distance_metric(desc_13, zeros(size(desc_13))) 

%blob1 and blob2 should be 3d matrices of the same size.
function dist = distance_metric(blob1, blob2)
    dist = sum(sum(sum(abs(blob1 - blob2))));

function show_desc(desc, figID, titleName)
    figure(figID)
    colormap(gray)
    im = sum(abs(desc), 3)';
    %im = max(abs(desc), [], 3)';
    imagesc(im);
    %imagesc(log(im / max(im(:))))
    colorbar
    title(titleName)

function desc = AlexNet_descriptor(myWindow)

    [h w d] = size(myWindow);
    im = single(myWindow);
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]); % - IMAGE_MEAN;

%    im(:,:,1) = im(:,:,1) - 104.00; % B imagenet mean
%    im(:,:,2) = im(:,:,2) - 116.66; % G
%    im(:,:,3) = im(:,:,3) - 122.67; % R
    im = permute(im, [2,1,3]); %[h w d] -> [d h w]

    desc = caffe('forward', {im});
    %returns flattened descriptor (your job to reshape it)
 
