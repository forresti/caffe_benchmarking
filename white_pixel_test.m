
function white_pixel_test()
    img = single(zeros(227, 227, 3));
    img(100, 100, :) = 255.0; %one white pixel
    img(200, 200, :) = 255.0; %one white pixel

    figure(1)
    imshow(img);
    title('original image')

    AlexNet_setup();
    alexnet_desc{1} = AlexNet_descriptor(img); 

    figure(4)
    colormap(gray)
    %imagesc(sum(abs(alexnet_desc{1}), 3)')
    im = max(abs(alexnet_desc{1}), [], 3)';
    imagesc(log(im / max(im(:))))
    colorbar
    title('conv5 max')

%227x227, 1 window per batch.
function AlexNet_setup()
    %model_def_file = '../examples/imagenet_deploy.prototxt';
    model_def_file = '../../examples/imagenet_deploy_batchsize1_output_conv5.prototxt';
    %model_def_file = '../../examples/imagenet_deploy_batchsize1_output_pool2.prototxt';
    %model_def_file = '../../examples/imagenet_deploy_batchsize1_output_conv1.prototxt'; %227x227 -> 55x55
    %model_def_file = '../examples/imagenet_deploy_batchsize1_input_500x500_output_conv1.prototxt'; %500x500 -> 123x123 
    %model_def_file = '../../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv1.prototxt';
    model_file = '../../examples/alexnet_train_iter_470000';
    caffe('init', model_def_file, model_file);
    caffe('set_mode_gpu');

function desc = AlexNet_descriptor(myWindow)

    [h w d] = size(myWindow);
    %assert( (h==227) && (w==227) );

    % resize to fixed input size
    im = single(myWindow);
    im = imresize(im, [227 227], 'bilinear');
    %im = imresize(im, [500 500], 'nearest');
    %im = imresize(im, [2000 2000], 'nearest');
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]); % - IMAGE_MEAN;

%    im(:,:,1) = im(:,:,1) - 104.00; % B imagenet mean
%    im(:,:,2) = im(:,:,2) - 116.66; % G
%    im(:,:,3) = im(:,:,3) - 122.67; % R
    im = permute(im, [2,1,3]); %[h w d] -> [d h w]

    scores = caffe('forward', {im});
    desc = reshape(scores{1}, [13 13 256]); %conv5
    %desc = reshape(scores{1}, [55 55 96]); %conv1
    %desc = reshape(scores{1}, [123 123 96]); %conv1, with 500x500
    %desc = reshape(scores{1}, [498 498 96]); %conv1, with 2000x2000
    

