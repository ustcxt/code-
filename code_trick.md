<big>在caffe中如何指定gpu:

    import caffe
	GPU_ID = 1 # Switch between 0 and 1 depending on the 	GPU you want to use.
	caffe.set_mode_gpu()
	caffe.set_device(GPU_ID)


python中文件读写操作
	
    data = {"position":"A","pocket":"B"}
    import pickel
    #文件写操作
    save_file = open("save.dat",'wb')
    pickle.dump("data",save_file)
    save.file.close()
    # 文件读操作
    read_file = open("save.dat","rb")
    filr_load=pickle.load(read_file)
    read_file.close()
    print(file_load)
    
   简单配置vim
   	
    vim ~/.vimrc
    输入：
    set ts=4
    set nu
   
   将TensorFlow的提示等级降低
   
	vim  ~/.bashrc
    export TF_CPP_MIN_LOG_LEVEL=2
	source ~/.bashrc
   
   tf.get_collection(" ")
   	
    #从集合中取出全部变量，生成一个列表
    
   tf.add_n([])
    
    #列表内对应元素相加
   tf.cast(x,dtype)
   
    #把x转化为dtype类型
    
   tf.argmax(x,axis)
   
    #返回最大值所在索引号，如：tf.argmax([[1,0,0]],1) 返回0
    
   os.path.join("home","name")
   
    #返回home/name
   字符串.split()
   	#按指定拆分符对字符串切片，返回分割后的列表，如：“./model/mnist_model-1001”.split("/")[-1].split("-")[-1] 返回1001
    
   with tf.Graph().as_default() as g:
   	#其内定义的节点在计算图g中
    
   保存模型
   
    saver = tf.train.Saver()  # 实例化saver对象
    with tf.Session() as sess:
    	for i in range(STEPS):
        	if i% 轮数 == 0：
            	saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

加载模型：
	
    with tf.Session() as sess:
    	ckpt = tf.train.get_checkpoint_state(存储路径)
        if ckpt and ckpt.model_checkpoint_path:
        	saver.restore(sess,ckpt.model_checkpoint_path)
 
实例化可还原滑动平均值的saver

	ema = tf.train.ExponentialMovingAverage(滑动平均系数)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)
 
 准确率计算方法:
 
 	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
 	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float.32))
            
损失函数中加入正则化：
	
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))
在定义参数w时加入正则化：
	
    if regularizer!=None:
   		tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
使用指数衰减学习率：
	
    learning_rate = tf.train.exponential_decay(
    	LEARNING_RATE_BASE,
        global_step,
        LEARNING_RATE_STEP,  # 总样本数/batch_size   
        LEARNING_RATE_DECAT,
        staircase=True
        )
使用滑动平均：
	
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
    	train_op = tf.no_op(name='train')
  
 打乱txt文件中的行：
 	
    shuf filename -o changed_filename
实现断点续训：
	
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
    	saver.restore(sess,ckpt.model_checkpoint_path)
生成tfrecords文件：
	
    writer = tf.python_io.TFRecordWriter(tfRecordName) #新建一个writer
    for 循环遍历每张图片和标签:
    	example = tf.train.Example(features=tf.train.Features(feature={
			'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
			'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
			}))  #将每张图片和标签封装到example中
        writer.write(example.SerializeToString())    #把example 进行序列化
    writer.close()
    
解析tfrecords文件：

	filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader() #新建一个reader
    
	_,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
										features = {
											'label':tf.FixedLenFeature([10],tf.int64),
											'img_raw':tf.FixedLenFeature([],tf.string)
											})
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img.set_shape([784])
    img = tf.cast(img,tf.float32)*(1./255)
	label = tf.cast(features["label"],tf.float32)
    
    
   使tfrecord效率更快的方法：

	#生成tfrecords时
    img_raw = tf.gfile.FastGFile(img_path,'rb').read() #读取图片时使用该函数效率更快
    
    #读取tfrecords时
	img = features['img_raw']
	img = tf.image.decode_png(img,channels=1)
	#img.set_shape([784])
	img = tf.reshape(img,[784])
Tensorflow 中的卷积：
	
    tf.nn.conv2d(输入描述，[batch,5,5,1],
    			卷积核描述，[3,3,1,16],
                滑动步长，[1,1,1,1],
                padding="VALID")
 Tensorflow 中的池化：
 	
    pool = tf.nn.max_pool(输入描述：[bacth,28,28,6],
    						池化核描述：[1,2,2,1],
                            滑动步长：[1,2,2,1],
                            padding = 'SAME')
    pool = tf.nn.avg_pool()与之相同
    
 Tensorflow 中的Dropout：
 
 	tf.nn.dropout(上层输出，暂时舍弃的概率)
    在使用时，一般仅在训练过程中使用dropout：
    如下：
    if train:
    	输出=tf.nn.dropout(上层输出，暂时舍弃的概率)
      
   np.load()和np.save()
   	
    #将数据以二进制格式读出/写入磁盘，扩展名为.npy
    np.save("name.npy",某数组)
    某变量 = np.load("name.npy",encoding={'latin1'or'ASCII'or'bytes'}).item()
    encoding项默认为‘ASCII’
    .item() 
    遍历其中键值对，导出模型参数赋值到前变量
    
 tf.shape(a)
 	
    #可返回a的维度
    a可以为tensor，list,array
tf.nn.bias_add(乘加和，bias)
	
    #把bias加到乘加和上
tf.reshape(tensor,[shape])
np.argsort(list)

	#对列表从小到大排列，返回索引值
os.getcwd()

	#返回当前工作目录
	用法可以：
    vgg16_path = os.path.join(os.getcwd(),"vgg16.npy")
    可返回  当前路径/vgg16.npy
  tf.split(切谁，怎么切，在哪个维度切)
  
	split0,split1,split2 = tf.split(value,[4,15,11],1)
    #value的值是一个5行20列的tensor，将其在第一个维度，切成3份，每份为[5,4],[5,15],[5,11]
    red,green,blue = tf.split(输入，3,3)
    
 tf.concat(值，在哪个维度)
 	
    t1 = [[1,2,3],[4,5,6]]
    t2 = [[7,8,9],[10,11,12]]
    tf.concat([t1,t2],0)
    # [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    tf.concat([t1,t2],1)
    # [[1,2,3,7,8,9],[4,5,6,10,11,12]]
    
 显示图像
 
 	fig = plt.figure("name")   #实例化图对象
    img = io.imread(图片路径)  #读入图片
    ax = fig.add_subplot(数 数 数)  #行 列 第几个
    ax.bar(bar的个数，bar的值，每个bar的名字，bar宽，bar色)   #柱状图
    ax.set_ylabel("") # y轴名字 u“中文”
    ax.set_title("")
	ax.text(文字x坐标，y坐标，内容，ha=‘center’，va=‘bottom’，fontsize=7)  # 横向，纵向 字号
    ax = imshow(图)  #画子图
 tf.constant()函数
 	
    # 括号内可以是字符串，或数字
    hello = tf.constant('Hello, TensorFlow!')
    a = tf.constant(2)
	b = tf.constant(3)
 查看当前目录下user目录的大小，并不想看其他目录以及其子目录：
 
 	du -sh user
	 -s表示总结的意思，即只列出一个总结的值
	 du -h --max-depth=0 user
	 --max-depth=n表示只深入到第n层目录，此处设置为0，即表示不深入到子目录。
     
 ubuntu 恢复误删除的文件
 	
    cd ~/.local/share/Trash/files/
    #该目录下有被删除的文件
 