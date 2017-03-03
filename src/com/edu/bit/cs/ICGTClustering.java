package com.edu.bit.cs;

import java.util.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import javax.swing.JFrame;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.datanucleus.store.types.backed.*;

public class ICGTClustering
{
	//树的根节点
	private ICGTNode _nodeRoot;

	//树的全部叶子结点
	private ArrayList<ICGTNode> _nodesLeaf;

	//一些运算里用到的常量值
	private static final double ZERO = 0.000001;
	private static final double G_CONNECT_THRESHOLD = 3.7;
	private static final int DISTANCE_THRESHOLD = 0;
	private static final double GMM_CONNECT_THRESHOLD = 2;

	public ICGTClustering()
	{
		_nodesLeaf = new ArrayList<ICGTNode>();
		_nodeRoot = null;
	}

	//接受一个新的RDD，用于增量聚类（先把数据转化为gmm,然后再把每个高斯模型插入高斯混合模型树里，再进行树的更新）
	public ICGTClustering run(JavaRDD<Vector> samples) throws Exception
	{
		if(samples.count() == 0)
		{
			return this;
		}

		//得到GMM模型
		int k = ensureClusterNum(samples.count());
		ICGTGaussianMixtureModel icgtGMM = MathUtil.getGMMByEM(samples, k);

		insertGMMToTree(icgtGMM);
		return this;
	}

	//将新得到的GMM插入ICGT树中并进行树的更新
	private void insertGMMToTree(ICGTGaussianMixtureModel icgtGMM) throws Exception
	{
		//比较每个高斯模型与叶子结点的距离，如果大于阈值，就先建结点，否则插入到最小的距离的叶子结点里
		for(int i = 0; i < icgtGMM.numOfGaussians() ; i++)
		{
			System.out.println("numOfGaussians:" + i + " / "+ icgtGMM.numOfGaussians() );

			//寻找最近叶子节点
			double minDistance = Double.MAX_VALUE;
			ICGTNode leafNearest = null;
			for(int j = 0; j < _nodesLeaf.size() ; ++j)
			{
				double distance = MathUtil.eulcideanDistance( icgtGMM.gaussian(i), _nodesLeaf.get(j).getGMM().gaussian(0));
				if(distance < minDistance)
				{
					minDistance = distance;
					leafNearest = _nodesLeaf.get(j);
				}
			}

			//说明是树的第一个结点
			if(minDistance == Double.MAX_VALUE)
			{
				_nodeRoot = new ICGTNode();
				_nodeRoot.initialize(ICGTNode.NODE_TYPE.ROOT);

				ICGTNode leafNew = new ICGTNode();
				leafNew.initialize(ICGTNode.NODE_TYPE.LEAF);

				//高斯模型对应于树结点的ID
				leafNew.setGMM(icgtGMM);

				_nodesLeaf.add(leafNew);
				_nodeRoot.addChild(leafNew);

				_nodeRoot.updateWeightOfChildren();
				_nodeRoot.mergeGuassians();

				//updateMuSigma(newLeaf);//更新树结点的均值和协方差矩阵
				//modify1(root);
			}
			//小于阈值，则将信息导入
			else if(minDistance < DISTANCE_THRESHOLD)
			{
				 leafNearest.insertGaussian(icgtGMM.gaussian(i), icgtGMM.numOfSamples(i));

				 leafNearest.getNodeFather().updateWeightOfChildren();
				 //updateMuSigma(minLeaf);//更新树结点的均值和协方差矩阵
			}
			//生成一个新的叶子结点
			else
			{
				System.out.println("生成叶子节点开始:" );
				ICGTNode leafNew = new ICGTNode();
				leafNew.initialize(ICGTNode.NODE_TYPE.LEAF);

				leafNew.setGMM(icgtGMM);
				leafNew.addData(icgtGMM.gaussian(0).mu());

				leafNearest.getNodeFather().addChild(leafNew);
				_nodeRoot = leafNew.getNodeFather().update();
				_nodesLeaf.add(leafNew);
				System.out.println("生成叶子节点结束:" );
				//updateMuSigma(newLeaf);//更新树结点的均值和协方差矩阵
			}
			System.out.println("_nodesLeaf.size():" + i + " / "+ _nodesLeaf.size());
		}
	}


	//根据item的数目来确定k
	private int ensureClusterNum(long num)
	{
		return (int)(num);
	}


	/****************************************************************************
	 BFSGetBestCluster:计算根节点下所有聚类组合中IQ\EQ值最小的聚类组合
	 输入：
	 @root:根节点
	 @bestCluster:存放最佳聚类结果的容器
	 ****************************************************************************/
	public LinkedList<ICGTNode> getBestCluster()
	{
		LinkedList<ICGTNode> queueNodes = new LinkedList<ICGTNode>();
		LinkedList<ICGTNode> bestClusters = new LinkedList<ICGTNode>();
		ICGTNode nodeIt = _nodeRoot.getNodeChild();
		while(nodeIt != null)
		{
			queueNodes.offer(nodeIt);
			nodeIt = nodeIt.getNodeBrotherNext();
		}

		double minRatio = Double.MAX_VALUE;

		while (queueNodes.size() != 0)
		{
			System.out.println("queueNodes.size():"+queueNodes.size());
			double ratio = clusteringQuality(queueNodes);
			System.out.println("clusteringQuality(queueNodes):"+ratio);
			if (ratio < minRatio)
			{
				minRatio = ratio;
				bestClusters = new LinkedList<ICGTNode>(queueNodes);
			}

			nodeIt = queueNodes.poll().getNodeChild();
			if(nodeIt.isLeaf())
			{
				break;
			}
			while(nodeIt != null)
			{
				queueNodes.offer(nodeIt);
				nodeIt = nodeIt.getNodeBrotherNext();
			}
		}
		return bestClusters;
	}



	/****************************************************************************
	 countRation1:计算某种聚类结果的IQ\EQ
	 输入：
	 @cluster:存放聚类结果的容器
	 ****************************************************************************/
	private double clusteringQuality(LinkedList<ICGTNode> clusters)
	{
		double IQ = 0, EQ = 0;
		Iterator<ICGTNode> iIt = clusters.iterator();
		while(iIt.hasNext())
		{
			System.out.println("clusteringQuality");
			//簇内距离
			ICGTNode iNode = iIt.next();
			double temp = calculateIQ(iNode);
			IQ += temp;

			//簇间距离
			Iterator<ICGTNode> jIt = clusters.iterator();
			while(jIt.hasNext() && !jIt.next().equals(iNode));
			while(jIt.hasNext())
			{
				EQ += calculateEQ(iNode,jIt.next());
			}
		}
		return IQ / EQ;
	}

	/****************************************************************************
	 countIQ:计算某种聚类结果的IQ值
	 输入：
	 @node:计算该节点下所有高斯成分的IQ
	 ****************************************************************************/
	private double calculateIQ(ICGTNode node)
	{
		System.out.println("clusteringQuality");
		ArrayList<ICGTNode> children = node.getChildren();

		double max = ZERO;
		for (int i = 0; i < children.size(); ++i)
		{
			double min = Double.MAX_VALUE;
			for (int j = i + 1; j < children.size(); ++j)
			{
				double temp = MathUtil.GQFDistance(children.get(i).getGMM(), children.get(j).getGMM());
				min = temp < min ? temp : min;
			}
			max = min > max ? min : max;
		}

		return max;
	}

	/****************************************************************************
	 countEQ:计算两个节点的EQ
	 输入：
	 @node1:节点1
	 @node2:节点2
	 ****************************************************************************/
	private double calculateEQ(ICGTNode nodeA, ICGTNode nodB)
	{
		System.out.println("calculateEQ");
		return MathUtil.GQFDistance(nodeA.getGMM(), nodB.getGMM());
	}

	//通过修改showResults可以修改聚类策略和显示形式
	public void showResults(JavaRDD<Vector> samples)
	{
		List<Integer> listLable = _nodeRoot.predict(samples);
		if(listLable == null){
			listLable = new ArrayList<>();
		}
		int i = 0;
		for(Integer label: listLable)
		{
			System.out.println("（）----->" + label);
		};
		/*
		for(int i = 0; i < _nodesLeaf.size(); ++i)
		{
			System.out.println("第"+i+"个叶子的均值为"+_nodesLeaf..gaussians()[0].mu());
			System.out.println("第"+i+"个叶子的协方差矩阵为"+_nodesLeaf.get(i).gmm.gaussians()[0].sigma());
		}*/

		final JFrame frame = new JFrame("Point Data Rendering");
		ICGTPanel panel = new ICGTPanel();
		panel.displayClusters(samples.collect(),listLable);
		frame.setContentPane(panel);
		frame.pack();
		frame.setVisible(true);
		frame.repaint();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public void getClusteredSamples(LinkedList<Vector> listSamples, LinkedList<Integer> listLable,ICGTNode node, int label)
	{
		if(node.isLeaf())
		{
			Iterator it = node.getData().iterator();
			while(it.hasNext())
			{
				listSamples.offer((Vector) it.next());
				listLable.offer(label);
			}
		}

		ICGTNode children = node.getNodeChild();
		while(children != null)
		{
			getClusteredSamples(listSamples, listLable, node, label);
			children = children.getNodeBrotherNext();
		}
	}

	public void showResults()
	{
		LinkedList<Vector> listSamples = new LinkedList<Vector>();
		LinkedList<Integer> listLable = new LinkedList<Integer>();

		LinkedList clusters = this.getBestCluster();

		Iterator<ICGTNode> iIt = clusters.iterator();
		int label = 0;
		while(iIt.hasNext())
		{
			getClusteredSamples(listSamples,listLable,iIt.next(),label++);
		}

		final JFrame frame = new JFrame("Point Data Rendering");
		ICGTPanel panel = new ICGTPanel();
		panel.displayClusters(listSamples,listLable);
		frame.setContentPane(panel);
		frame.pack();
		frame.setVisible(true);
		frame.repaint();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
}
