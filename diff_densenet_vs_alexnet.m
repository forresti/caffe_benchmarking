
function diff_densenet_vs_alexnet()
    cls = 'bicycle';
    conf = voc_config();
    cachedir = conf.paths.model_dir;
    [pos, neg, impos] = pascal_data(cls, conf.pascal.year);
    %templateSize = [13 13]; %typical Alexnet conv5 size.
    templateSize = [55 55]; %conv1
    %templateSize = [123 123]; %conv1, with 500x500 input to alexnet

    boxes(1) = to_bbox([1, 1, 227, 227]);
    boxes(2) = to_bbox([101, 101, 327, 327]);
    boxes(3) = to_bbox([50 50 200 200]);
    boxes(4) = to_bbox([50 200 200 350]);

    %DenseNet features
    my_DenseNet_setup();
imgId=3;
boxId=1;
    pyra_params.interval=10;
    pyra = convnet_featpyramid(pos(imgId).im, pyra_params);
    [densenet_desc{1}, scaleIdx, roundedBox_in_px] = get_featureSlice(pyra, boxes(boxId), templateSize);
roundedBox_in_px
scaleIdx
pyra
    figure(1)
    colormap(gray)
    %imagesc(sum(densenet_desc{1}, 3))
    imagesc(max(densenet_desc{1}, [], 3))
    colorbar
    title('densenet')
    %imagesc(sum(pyra.feat{5}, 3)) %no need to transpose with DenseNet.


    AlexNet_setup();
    img = imread(pos(imgId).im);
    %for i=1:length(boxes)
        myWindow = img(boxes(boxId).y1 : boxes(boxId).y2, boxes(boxId).x1 : boxes(boxId).x2, :);
        %myWindow = img;
        figure(4) 
        imshow(myWindow)
        title('crop that we fed to alexnet')
        alexnet_desc{1} = AlexNet_descriptor(myWindow); 
    %end

    figure(2)
    colormap(gray)
    alexnet_desc_trans = permute(alexnet_desc{1}, [2 1 3]); %rotate 90 degrees (row to col major)
    %imagesc(sum(alexnet_desc_trans, 3))
    imagesc(max(alexnet_desc_trans, [], 3))
    colorbar;
    title('alexnet')

    figure(3)
    imshow(img);
    title('original image')

    sum(sum(sum(abs(alexnet_desc_trans - densenet_desc{1})))) %difference

    keyboard

function box=to_bbox(arr)
    box.x1=arr(1);
    box.y1=arr(2);
    box.x2=arr(3);
    box.y2=arr(4);

%227x227, 1 window per batch.
function AlexNet_setup()
    %model_def_file = '../examples/imagenet_deploy.prototxt';
    %model_def_file = '../examples/imagenet_deploy_batchsize1_output_conv5.prototxt';
    model_def_file = '../examples/imagenet_deploy_batchsize1_output_conv1.prototxt'; %227x227 -> 55x55
    %model_def_file = '../examples/imagenet_deploy_batchsize1_input_500x500_output_conv1.prototxt'; %500x500 -> 123x123 
    %model_def_file = '../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv1.prototxt';
    model_file = '../examples/alexnet_train_iter_470000';
    caffe('init', model_def_file, model_file);
    caffe('set_mode_gpu');

function my_DenseNet_setup()
    %model_def_file = '../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv5.prototxt'
    model_def_file = '../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv1.prototxt'
    model_file = '../examples/alexnet_train_iter_470000'; % NOTE: you'll have to get the pre-trained ILSVRC network
    %model_file = '../examples/finetune_voc_2007_trainval_iter_70000'; %Ross -- fine-tuned on pascal trainval 
    caffe('init', model_def_file, model_file); % init caffe network (spews logging info)
    caffe('set_mode_gpu');
    caffe('set_phase_test');

function desc = AlexNet_descriptor(myWindow)

    %d = load('ilsvrc_2012_mean');
    %IMAGE_MEAN = d.image_mean;
    %IMAGE_DIM = 256;
    %CROPPED_DIM = 227;

    [h w d] = size(myWindow);
    %assert( (h==227) && (w==227) );

    % resize to fixed input size
    im = single(myWindow);
    im = imresize(im, [227 227], 'bilinear');
    %im = imresize(im, [500 500], 'nearest');
    %im = imresize(im, [2000 2000], 'nearest');
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]); % - IMAGE_MEAN;

    im(:,:,1) = im(:,:,1) - 104.00; % B imagenet mean
    im(:,:,2) = im(:,:,2) - 116.66; % G
    im(:,:,3) = im(:,:,3) - 122.67; % R
    im = permute(im, [2,1,3]); %[h w d] -> [d h w]

    scores = caffe('forward', {im});
    %desc = reshape(scores{1}, [13 13 256]); %conv5
    desc = reshape(scores{1}, [55 55 96]); %conv1
    %desc = reshape(scores{1}, [123 123 96]); %conv1, with 500x500
    %desc = reshape(scores{1}, [498 498 96]); %conv1, with 2000x2000
    

