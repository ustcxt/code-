## *code reading* 
***
## <center>**Single View Stereo Matching**<center\>
#### <center>Yue Luo 1 ∗ Jimmy Ren 1 ∗ Mude Lin 1 Jiahao Pang 1 Wenxiu Sun 1 Hongsheng Li 2 Liang Lin 1,3
#### <center>$^1$SenseTime Research
#### <center>$^2$The Chinese University of Hong Kong, Hong Kong SAR, China
#### <center>$^3$Sun Yat-sen University, China<center\>
#### <center>$^1${luoyue,rensijie,linmude,pangjiahao,sunwenxiu,linliang}@sensetime.com
#### <center>$^2$hsli@ee.cuhk.edu.hk
***
<big>	
**trainConfig.m**

    在/home/cxt/code/SVS/training/路径下存在 trainConfig.m 文件——caffe配置训练参数文件
    fprintf('Setting to GPU mode\n');
    caffe.set_mode_gpu();
	caffe.set_device(0);             %配置使用gpu模式，并且默认设置gpu_id号为0
    
    param.model(1).preModel = './prototxt/viewSynthesis_BN/preModel/VGG_ILSVRC_16_layers.caffemodel';
    %模式1的预训练模型为VGG16
    
    param.model(1).description = 'View Synthesis model trained on Kitti. Developed based on deep3D model, with BN layers';
    % 该模型的训练数据是KITTI，并且是有BN层的
    
    param.model(1).solverFile = './prototxt/viewSynthesis_BN/ViewSyn_BN_solver.prototxt';
    %配置优化文件路径
    
    param.model(1).trainFile = './prototxt/viewSynthesis_BN/ViewSyn_BN_train.prototxt';
    %训练文件，即网络模型为ViewSyn_BN_train.prototxt
    
    param.model(1).width = 640;       %设置输入图片宽
	param.model(1).height = 192;      %设置输入图片高
	param.model(1).batchSize = 2;     %batch size设置为2
	param.model(1).channel = 65;	  %视差通道数设置为65，即表示可能的视差范围为0-
    param.model(1).stride = 1;   %this control the disparity scale step. Max disparity will be 32*stride. So only 1/2 will be needed.% 暂时不明白
    
    param.model(2).preModel = './prototxt/viewSynthesis_BN/caffemodel/viewSyn_BN.caffemodel';
    % 模式2中使用的预训练模型是viewSyn_BN.caffemodel
    
    param.model(2).trainFile = './prototxt/viewSynthesis/ViewSyn_train.prototxt';
    %训练网络为ViewSyn_train.prototxt
    
    param.model(3).preModel1 ='./prototxt/viewSynthesis/caffemodel/viewSyn.caffemodel';     	%first stage network
	param.model(3).preModel2 ='./prototxt/stereo/caffemodel/disp_flyKitti.caffemodel';          %second stage network
    %模式3即联合训练
    
    param.model(3).trainFile1 = './prototxt/svs/svs_train.prototxt';
	param.model(3).trainFile2_tmp = './prototxt/svs/stereo_train_tmp.prototxt';
    %训练网络配置分别为....
    
    param.model(3).W = 1280;
	param.model(3).H = 384; %暂时没看明白
    
 **ViewSyn_BN_solver.prototxt**       	    
 
 	该配置文件主要设置优化方法及参数
    文件路径 /home/cxt/code/SVS/training/prototxt/viewSynthesis_BN/ViewSyn_BN_solver.prototxt
    net: "./prototxt/viewSynthesis_BN/ViewSyn_BN_train.prototxt"
	%The base learning rate, momentum and the weight decay of the network.
	base_lr: 0.0002
    %基础学习率
	momentum: 0.9
    %惯性
	weight_decay: 0.0005
    %权值衰减
	% The learning rate policy
	lr_policy: "step"
	gamma: 0.5
	stepsize: 80000
	% Display every 5 iterations
	display: 5
	% The maximum number of iterations
	max_iter: 100000
	% solver mode: CPU or GPU
	solver_mode: GPU
	solver_type: ADAM
	momentum2: 0.999
	delta: 1e-4
**train_viewSyn.m**

	文件路径 /home/cxt/code/SVS/training/train_viewSyn.m
    solver.net.copy_from(preModel);  %从模型中拷贝参数
    
    epoch = 200;
	for ep = 1:epoch
    	order_set = randperm(length(trainSets));
        %length(trainSets)=23，生成从1-23的随机且不重复的排列
    	for i = 1:length(trainSets)
        	clear train;
        	fprintf('Loading trainSet %s for Training....\n',trainSets(order_set(i)).name);
        	trainData  = load([trainAdd '/' trainSets(order_set(i)).name]);
        	if(strcmp(trainSets(order_set(i)).name,'trainKitti15.mat'))
            	train = trainData.gt;
            	seqLength = train.size(1);
			else
            	train = trainData.train;
            	seqLength = train.length;
        	end    
        	clear trainData;
            %随机加载训练数据
            order = randperm(seqLength);
        	for batch = 1:floor(seqLength/batch_size) 

            	data_ = zeros([dim(2) dim(1) 3 batch_size]);
            	label_ = zeros([dim(2) dim(1) 3 batch_size]);
            	shift = zeros([dim(2) dim(1) 3 batch_size channel]);
            	for n = 1:batch_size
                	trainInd = order( (batch-1)*batch_size + n);
 
                	[input,shiftInput,label] = transformation_viewSyn(train,trainInd,dim,channel,stride);
                    % 感觉是做数据增值
                	data_(:,:,:,n) = (input);
                	label_(:,:,:,n) = (label);
                	shift(:,:,:,n,:) =(shiftInput);
            	end
                solver.net.blobs('data').set_data(single(data_));
            	solver.net.blobs('label').set_data(single(label_));
            	for k =1:channel
               	ss = strcat('dis',num2str(k-1));
              	 solver.net.blobs(ss).set_data(single(shift(:,:,:,:,k)));
            	end    
            	clear input label shiftInput data_ label_ shift input;
           	 solver.step(1);
            	iter = solver.iter();   

            	%save the model
            	if(rem(iter,10000)==0)
               	 fprintf('Saving model for iter %d...\n',iter);
               	 solver.net.save([saveDir '/caffemodel/viewSyn_iter_' num2str(iter) '.caffemodel']);
           	 end  
             	%每10000次迭代保存一次模型
            
           	 %Save visual result
            	if(rem(iter,5000)==0 || iter == 1 )
                	data = solver.net.blobs('data').get_data();
                	label = solver.net.blobs('label').get_data();
                	pred = solver.net.blobs('pred_right').get_data();

                	idx = 1;

                	h=figure('Visible', 'off');hold on;subplot(1,3,1);imshow(uint8(recover(data(:,:,:,idx))),[]);subplot(1,3,2);imshow(uint8(recover(label(:,:,:,idx))),[]);subplot(1,3,3);imshow(uint8(recover(pred(:,:,:,idx))),[]);

                	saveas(h,strcat(saveDir,'/fig/figure_',num2str(iter),'.png'));
                	clf; clear data label pred;
            	end
            end
        end
	end
            

    
    
    
    
    