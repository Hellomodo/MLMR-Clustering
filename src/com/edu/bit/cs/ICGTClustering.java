package com.edu.bit.cs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.spark.mllib.linalg.*;

import java.util.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import javax.swing.JFrame;

public class ICGTClustering
{
	//树的根节点
	private ICGTNode _nodeRoot;

	private LinkedList<Sample> _queueSamples;
	//树的全部叶子结点
	private LinkedList<ICGTNode> _nodesLeaf;

	//一些运算里用到的常量值
	private static final double ZERO = 0.000001;
	private static final double DISTANCE_THRESHOLD = 0; //20000

	public ICGTClustering()
	{
		_nodesLeaf = new LinkedList<ICGTNode>();
		_queueSamples = new LinkedList<Sample>();
		_nodeRoot = null;
	}

	public ICGTClustering(Iterator<ICGTNode> itNode)
	{
		_nodesLeaf = new LinkedList<ICGTNode>();
		_queueSamples = new LinkedList<Sample>();
		_nodeRoot = null;

		System.out.println("Final reclustering");

		_nodeRoot = new ICGTNode(ICGTNode.NODE_TYPE.ROOT);
		while (itNode.hasNext())
		{
			ICGTNode subtree = itNode.next();
			subtree.isChanged(true);
			_nodeRoot.addChild(subtree);
		}

		ArrayList<ICGTNode> clusters = _nodeRoot.getNodesChildren();

		Vector< Vector<Double> > distance2d = new Vector< Vector<Double> >();
		for(int i = 0; i < clusters.size(); i++)
		{
			Vector<Double> distance1d = new Vector<Double>();
			for(int j = 0; j < clusters.size(); j++)
			{
				Double distance = MathUtil.KLDivergence(clusters.get(i).getGMM(), clusters.get(j).getGMM());
				distance1d.add(distance);
			}
			distance2d.add(distance1d);
		}

		while(true)
		{

			if(_nodeRoot.numOfChild() == 2)
				break;

			clusters = _nodeRoot.getNodesChildren();
			double minDistance = Double.MAX_VALUE;
			int iNode = -1, jNode = -1;

			for(int i = 0; i < clusters.size(); i++)
			{
				Vector<Double> distance1d =  distance2d.get(i);
				for(int j = 0; j < clusters.size(); j++)
				{
					double distance = distance1d.get(j);
					if(distance < minDistance && i != j)
					{
						minDistance = distance;
						iNode = i;
						jNode = j;
					}
				}
			}

			clusters.get(iNode).seperateFromParents();
			clusters.get(jNode).seperateFromParents();
			ICGTNode node = new ICGTNode(ICGTNode.NODE_TYPE.OTHER);
			node.addChild(clusters.get(iNode));
			node.addChild(clusters.get(jNode));
			node.mergeGuassians();
			_nodeRoot.addChild(node);
			_nodeRoot.mergeGuassians();

			if(iNode > jNode)
			{
				distance2d.remove(iNode);
				distance2d.remove(jNode);
				for(int i = 0; i < distance2d.size(); i++)
				{
					Vector<Double> distance1d =  distance2d.get(i);
					distance1d.remove(iNode);
					distance1d.remove(jNode);

				}
			}else{
				distance2d.remove(jNode);
				distance2d.remove(iNode);
				for(int i = 0; i < distance2d.size(); i++)
				{
					Vector<Double> distance1d =  distance2d.get(i);
					distance1d.remove(jNode);
					distance1d.remove(iNode);
				}
			}

			java.util.Vector distance1d = new java.util.Vector<Double>();
			clusters = _nodeRoot.getNodesChildren();
			for(int i = 0; i < clusters.size(); i++)
			{
				Double distance = MathUtil.KLDivergence(clusters.get(0).getGMM(), clusters.get(i).getGMM());
				distance1d.add(distance);
				if(i != 0)
				{
					distance2d.get(i - 1).add(0,MathUtil.KLDivergence(clusters.get(i).getGMM(), clusters.get(0).getGMM()));
				}
			}
			distance2d.add(0,distance1d);

		}

		clusters = _nodeRoot.getNodesChildren();
		for(int i = 0; i < clusters.size(); i++)
		{
			for(int j = 0; j < clusters.size(); j++)
			{
				System.out.println(i + ":" + j + "-->" + MathUtil.KLDivergence(clusters.get(i).getGMM(), clusters.get(j).getGMM()));
			}
		}

	}

	public List<ICGTNode> getFirstLayer()
	{
		ArrayList<ICGTNode> listNodes = new ArrayList<>();
		ICGTNode nodeIt = _nodeRoot.getNodeChild();
		while(nodeIt != null)
		{
			nodeIt.seperateFromParents();
			listNodes.add(nodeIt);
			nodeIt = nodeIt.getNodeBrotherNext();
		}
		System.out.println("num of listNodes" + listNodes.size());
		return listNodes;
	}


	//接受一个新的RDD，用于增量聚类（先把数据转化为gmm,然后再把每个高斯模型插入高斯混合模型树里，再进行树的更新）
	public ICGTClustering run(Iterator<org.apache.spark.mllib.linalg.Vector> samples)
	{
		if(samples == null)
		{
			return this;
		}

		long count = 0;
		while(samples.hasNext())
		{
            count ++;

			Sample sample = new Sample(samples.next());
			_queueSamples.offer(sample);

			GaussianMixtureModel gmmNew = new GaussianMixtureModel(new MultivariateGaussian(sample));

            System.out.println("寻找最近的叶子节点: "+ count + "/" );
			//寻找最近叶子节点
			double minDistance = Double.MAX_VALUE;
			ICGTNode leafNearest = null;
			Iterator<ICGTNode> itNode = _nodesLeaf.iterator();
			while (itNode.hasNext())
			{
				ICGTNode nodeTmp = itNode.next();
				double distance = MathUtil.eulcideanDistance( gmmNew.gaussian(0), nodeTmp.getGMM().gaussian(0));
				if(distance < minDistance)
				{
					minDistance = distance;
					leafNearest = nodeTmp;
				}
			}

			//说明是树的第一个结点
			if(minDistance == Double.MAX_VALUE)
			{
				_nodeRoot = new ICGTNode(ICGTNode.NODE_TYPE.ROOT);

				ICGTNode leafNew = new ICGTNode(ICGTNode.NODE_TYPE.LEAF);

				//高斯模型对应于树结点的ID
				leafNew.setGMM(gmmNew);
				leafNew.addSample(sample);
				_nodesLeaf.add(leafNew);
				_nodeRoot.addChild(leafNew);
				_nodeRoot.mergeGuassians();
			}
			//小于阈值，则将信息导入
			else if(minDistance < DISTANCE_THRESHOLD)
			{
				leafNearest.setGMM(new GaussianMixtureModel(MathUtil.mergeGaussians(leafNearest.getGMM().gaussian(0),gmmNew.gaussian(0))));
				leafNearest.addSample(sample);
				_nodeRoot = leafNearest.getNodeFather().update();

			}
			//生成一个新的叶子结点
			else
			{
				ICGTNode leafNew = new ICGTNode(ICGTNode.NODE_TYPE.LEAF);

				leafNew.setGMM(gmmNew);
				leafNew.addSample(sample);

				leafNearest.getNodeFather().addChild(leafNew);
				_nodeRoot = leafNew.getNodeFather().update();
 				_nodesLeaf.add(leafNew);

			}
            System.out.println("插入叶子节点");
		}
		return this;
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
			if(nodeIt == null || nodeIt.isLeaf())
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
			//簇内距离
			ICGTNode iNode = iIt.next();
			double temp = calculateIQ(iNode);
			double tmp = IQ + temp;
            IQ = tmp;
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
		ArrayList<ICGTNode> children = node.getNodesChildren();

		double max = ZERO;
        int num =  children.size() - 1;
        for (int i = 0; i < num - 1; ++i)
		{
			double min = Double.MAX_VALUE;
			for (int j = i + 1; j < num; ++j)
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
		return MathUtil.GQFDistance(nodeA.getGMM(), nodB.getGMM());
	}

	public void getClusteredSamples(LinkedList<Sample> listSamples, ICGTNode node, int label)
	{
		if(node.isLeaf())
		{
			Iterator it = node.getSample().iterator();
			while(it.hasNext())
			{
                Sample sample = (Sample) it.next();
                sample.setLabel(label);
				listSamples.offer(sample);
			}
			return ;
		}

		ICGTNode children = node.getNodeChild();
		while(children != null)
		{
			getClusteredSamples(listSamples, children, label);
			children = children.getNodeBrotherNext();
		}
	}

	public void showResults()
	{
		LinkedList<Sample> listSamples = new LinkedList<Sample>();

		ArrayList<ICGTNode> clusters = _nodeRoot.getNodesChildren();
		;
		for(int label = 0; label < clusters.size(); label ++)
		{
			getClusteredSamples(listSamples, clusters.get(label), label);
		}

		final JFrame frame = new JFrame("Point Data Rendering");
		ICGTPanel panel = new ICGTPanel();
		panel.displayClusters(listSamples);
		frame.setContentPane(panel);
		frame.pack();
		frame.setVisible(true);
		frame.repaint();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
}
