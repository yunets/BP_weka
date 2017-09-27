/**
 *Project Name: BP_weka
 *File Name:    BPTest.java
 *Package Name: lys.bp
 *Date:         2017年9月27日 下午7:50:45
 *Copyright (c) 2017,578888218@qq.com All Rights Reserved.
*/

package lys.bp;

import java.io.File;
import java.io.IOException;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *Title:      BPTest<br/>
 *Description:
 *@Company:   励图高科<br/>
 *@author:    刘云生
 *@version:   v1.0
 *@since:     JDK 1.8.0_40
 *@Date:      2017年9月27日 下午7:50:45 <br/>
*/
public class BPTest {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//便于测试，用数组保存一些数据，从数据库中取数据是同理的
				//二维数组第一列表示当月的实际数据，第二列是上个月的数据，用于辅助对当月数据的预测的
				//二维数组的数据用于测试集数据，为了展示两种weka载入数据的方法，将训练集数据从arff文件中读取
				double[][] a = {{476005046,306349941},{377331965,476005046}}; 
				//double[][] a = {{4,3},{9,8}}; 
				
				//读入训练集数据
				File inputFile = new File("data\\gupiao.arff");//将路径替换掉
				ArffLoader atf = new ArffLoader();
				try {
					atf.setFile(inputFile);
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				Instances instancesTrain = atf.getDataSet();
				instancesTrain.setClassIndex(0);//设置训练数据集的类属性，即对哪个数据列进行预测（属性的下标从0开始）
				
				
				//读入测试集数据
				FastVector attrs = new FastVector();
				Attribute ratio = new Attribute("CURtrdsum",1);//创建属性，参数为属性名称和属性号，但属性号并不影响FastVector中属性的顺序
				Attribute preratio = new Attribute("PREtrdsum",2);
				attrs.addElement(ratio);//向FastVector中添加属性，属性在FastVector中的顺序由添加的先后顺序确定。
				attrs.addElement(preratio);
				Instances instancesTest = new Instances("bp",attrs,attrs.size());//创建实例集，即数据集，参数为名称，FastVector类型的属性集，以及属性集的大小（即数据集的列数）
				instancesTest.setClass(ratio);//设置数据集的类属性，即对哪个数据列进行预测
				for(int k=0;k<2;k++){
				Instance ins = new Instance(attrs.size());//创建实例，即一条数据
				ins.setDataset(instancesTest);//设置该条数据对应的数据集，和数据集的属性进行对应
				ins.setValue(ratio, a[k][0]);//设置数据每个属性的值
				ins.setValue(preratio, a[k][1]);
				instancesTest.add(ins);//将该条数据添加到数据集中
				}
				
				
				MultilayerPerceptron m_classifier = new MultilayerPerceptron();//创建算法实例，要使用其他的算法，只用把类换做相应的即可
				m_classifier.setAutoBuild(true);
				//m_classifier.setHiddenLayers("a");
				//m_classifier.setLearningRate(0.001);
				try {
			 		m_classifier.buildClassifier(instancesTrain); //进行训练
				} catch (Exception e) {
			 		e.printStackTrace();
			 	}
			
				for(int i = 0;i<2;i++){//测试分类结果
				 	//instancesTest.instance(i)获得的是用模型预测的结果值，instancesTest.instance(i).classValue();获得的是测试集类属性的值
					//此处是把预测值和当前值同时输出，进行对比
			 	try {
			 		System.out.println(m_classifier.classifyInstance(instancesTest.instance(i))+","+instancesTest.instance(i).classValue());
			 	}catch (Exception e) {
			 		e.printStackTrace();
			 	}
				}
				
				System.out.println("bp success!");
				System.out.println(m_classifier);
	}

}

