# chinese_bilstm_cnn_crf

## 0. 效果展示  

## 1. 安装相关文件  
* 人民日报词库下载私人地址  
  私人地址: 链接: <https://pan.baidu.com/s/13Zl7R_QLj-_cepwtuigyUg> 密码: 9yqf  
* 安装gensim-3.4.0.tar.gz文件,制作word2vec  
  官方链接: <https://pypi.org/project/gensim/#files>  
  私人地址(辅以相关依赖文件): 链接: <https://pan.baidu.com/s/1w3FTnhwzU_6i4-KUkYlIdg>  密码: pirg  
* 模型搭建第三方库Keras-2.1.6.tar.gz  
  私人地址: 链接: <https://pan.baidu.com/s/1ypoEgf6ITjcNalzTRtnvmw> 密码: uot8  
* 模型搭建第三方库keras-contrib-master.zip  
  私人地址: 链接: <https://pan.baidu.com/s/16D3nntBpDVVDqV1B66f7_Q> 密码: ibze  
* 绘制模型结构图所需文件graphviz-2.38.rar  
  官方下载链接: <https://graphviz.gitlab.io/download/>  
  私人地址: 链接: <https://pan.baidu.com/s/1ZxSE96upkPLkhjeaam9FIQ> 密码: 15v6  
* 绘制模型结构图所需文件pydot-1.2.4.tar.gz  
  私人地址: 链接: <https://pan.baidu.com/s/1xnv6TfpUBWAKbLrDYQL0Lg> 密码: gvt3  
  
## 2. 传统方法参考链接  
* 基于检索的方法mmseg: <http://technology.chtsai.org/mmseg/>  
* 基于统计的方法 Stanford Word Segmenter: <https://nlp.stanford.edu/software/segmenter.shtml>  

## 3. 执行命令  
* 3.0 下载词库文件解压放置在文件夹corpus中  
* 3.1 生成词向量模型model_vector_people.m  
  `python embedding_model.py`  
* 3.2 执行train.py文件训练模型  
  `python train.py`  
* 3.3 特殊函数说明(train.py)  

  3.3.1 create_label_data(word_dict, raw_train_file)--->创建train.data文件  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;人	B  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;民	M  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;网	E  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一	B  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;月	M  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一	M  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;日	E  
  
  3.3.2 documents_length = create_documents()--->创建data.data和label.data文件  
        data.data  
        人	民	网	一	月	一	日	讯	据	纽	约	时	报	报	道	，  
        美	国	华	尔	街	股	市	在	二	零	一	三	年	的	最	后	一	天	继	续	上	涨	，  
        和	全	球	股	市	一	样  
        label.data  
        B	M	E	B	M	M	E	S	S	B	E	B	E	B	E	S  
        B	E	B	M	E	B	E	S	B	M	M	M	E	S	B	E	B	E	B	E	B	E	S  
        S	B	E	B	E	B	E	S  
  
  3.3.3 lexicon, lexicon_reverse = create_lexicon(word_dict)--->创建lexicon.pkl文件  
        {'这': 75, '云': 307, '伏': 92, '共': 139, '问': 140, '跑': 308...}  
  
  3.3.4 create_matrix(lexicon, label_2_index)--->创建data_index.data和label_index.data文件  
        data_index.data  
        11	14	118	2	39	2	8	172	102	295	293	131	30	30	29	1  
        117	12	284	47	212	76	56	7	13	19	2	16	5	3	61	75	2	459	127	79	46	93	1  
        6	111	336	76	56	2	208	1  
        label_index.data  
        1	2	3	1	2	2	3	4	4	1	3	1	3	1	3	4  
        1	3	1	2	3	1	3	4	1	2	2	2	3	4	1	3	1	3	1	3	1	3	4  
        4	1	3	1	3	1	3	4  
  
