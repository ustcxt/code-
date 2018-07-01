# code reading

#### <center>Depth Map Prediction from a Single Image using a Multi-Scale Deep Network<center/>
 ##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;David Eigen  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Christian Puhrsch   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Rob Fergus

##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;deigen@cs.nyu.edu  &emsp;&emsp;&emsp;&emsp;&emsp;  cpuhrsch@nyu.edu   &emsp;&emsp;&emsp; &emsp;&emsp;&emsp; fergus@cs.nyu.edu
<br><big><big>
**convert_mat_to_img.py**

	import os
	import numpy as math_library
	import h5py 
	from PIL import Image as image_library
    
    train_file_path = './data/train.csv'    #存放训练数据集的路径
    
    def convert_nyu_dataset_into_images_and_csv(path):
    	print("loading dataset: %s" % (path))
    	nyu_dataset_file  = h5py.File(path)  #读取数据集
    	
        trainings_material = []  #空列表，之后存放原图-深度图像对
        for i, (image_original, image_depth) in enumerate(zip(nyu_dataset_file['images'], nyu_dataset_file['depths'])):
       	 image_original_transpose = image_original.transpose(2, 1, 0)  #将原始图像转置
        	image_depth_transpose = image_depth.transpose(1, 0)     # 将深度图像转置
        	image_depth_transpose = (image_depth_transpose/math_library.max(image_depth_transpose))*255.0     #将深度图归一化到0-255
        	image_original_pil = image_library.fromarray(math_library.uint8(image_original_transpose))   #将原始的矩阵转化成图片
        	image_depth_pil = image_library.fromarray(math_library.uint8(image_depth_transpose))    # 将原始深度转化为图片
        	image_original_name = os.path.join("/gdata", "chenxt","nyu_datasets", "%05d.jpg" % (i))  #原始图片的名字及路径
        	image_original_pil.save(image_original_name)
        #保存在该路径下
        	image_depth_name = os.path.join("/gdata", "chenxt","nyu_datasets", "%05d.png" % (i)) # 深度图片的名字和路径
        	image_depth_pil.save(image_depth_name) # 保存在该路径下
        	trainings_material.append((image_original_name, image_depth_name)) # 将-原始数据的名字和深度数据的名字组成一个个元组，放在空列表中
		write_csv_file(trainings_material)  #将上述列表存入csv文件中
        
	def write_csv_file(trainingsmaterial):
    	with open(train_file_path, 'w') as output:
        	for (image_original_path, image_depth_path) in trainingsmaterial:
            output.write("%s,%s" % (image_original_path, image_depth_path))
            output.write("\n")   #写如csv文件
            
	if __name__ == '__main__':
    	current_directory = os.getcwd()
    	nyu_dataset_path = '/gdata/chenxt/nyu_depth_v2_labeled.mat'
    	convert_nyu_dataset_into_images_and_csv(nyu_dataset_path)      #主函数
        
        
 **model.py**
 		
	import tensorflow as tf
	import math
	from model_part import conv2d
	from model_part import fullyConnectedLayer
    ## 粗网络
    def globalDepthMap(images, reuse=False, trainable=True):
    	with tf.name_scope("Global_Depth"):
        	coarse1_conv = conv2d('coarse1', images, [11, 11, 3, 96], [96], [1, 4, 4, 1], padding='VALID', reuse=reuse, trainable=trainable)
        	#coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        	coarse1 = tf.nn.max_pool(coarse1_conv,ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        	coarse2_conv = conv2d('coarse2', coarse1, [5, 5, 96, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
        	coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        	coarse3 = conv2d('coarse3', coarse2, [3, 3, 256, 384], [384], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
        	coarse4 = conv2d('coarse4', coarse3, [3, 3, 384, 384], [384], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
        	coarse5 = conv2d('coarse5', coarse4, [3, 3, 384, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
        	coarse6 = fullyConnectedLayer('coarse6', coarse5, [8*6*256, 4096], [4096], reuse=reuse, trainable=trainable)
        	coarse6_dropout = tf.nn.dropout(coarse6, 0.8)
        	coarse7 = fullyConnectedLayer('coarse7', coarse6_dropout, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
        	coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])
        return coarse7_output
        
     ## 细网络
 	def localDepthMap(images, coarse7_output, keep_conv, reuse=False, trainable=True):
    	with tf.name_scope("Local_Depth"):
        	fine1_conv = conv2d('fine1', images, [9, 9, 3, 63], [63], [1, 2, 2, 1], padding='VALID', reuse=reuse, trainable=trainable)
        	fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='fine_pool1')
        	#fine1_dropout = tf.nn.dropout(fine1, keep_conv)
        	fine2 = tf.concat(axis=3, values=[fine1, coarse7_output], name="fine2_concat")
        	fine3 = conv2d('fine3', fine2, [5, 5, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
        	#fine3_dropout = tf.nn.dropout(fine3, keep_conv)
        	#print("fine3_dropout ", fine3_dropout._shape)
        	print("fine3", fine3._shape)
        	fine4_conv = conv2d('fine4_conv', fine3, [5, 5, 64, 1], [1], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
        	print("fine4_conv ", fine4_conv._shape)
        	fine4_full = fullyConnectedLayer('fine4_full', fine4_conv, [55*74*1, 4070], [4070], reuse=reuse, trainable=trainable)
        	print("fine4_full ", fine4_full._shape)
        	fine4 = tf.reshape(fine4_full, [-1, 55, 74, 1])

        	#print("fine1_conv ", fine1_conv._shape)
        	#print("fine1 ", fine1._shape)
       	 #print("fine1_dropout ", fine1_dropout._shape)
        	#print("fine2 ", fine2._shape)
       	 #print("fine3 ", fine3._shape)
      	  #print("fine3_dropout ", fine3_dropout._shape)
       	 #print("fine4 ", fine4._shape)
        
    	#return fine4, fine3_dropout, fine3, fine2, fine1_dropout, fine1, fine1_conv
    	return fine4
    
    
**task.py**    

	def train():
    	with tf.Graph().as_default():
        	dataset = DataSet(BATCH_SIZE)
            input_images, depth_maps, depth_maps_sigma = dataset.create_trainingbatches_from_csv(TRAIN_FILE)  # rename variables
            test_image = testdata.load_test_image(TEST_FILE)   #记载测试图像
        
        
        
        
        