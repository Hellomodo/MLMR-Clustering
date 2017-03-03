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
	//���ĸ��ڵ�
	private ICGTNode _nodeRoot;

	//����ȫ��Ҷ�ӽ��
	private ArrayList<ICGTNode> _nodesLeaf;

	//һЩ�������õ��ĳ���ֵ
	private static final double ZERO = 0.000001;
	private static final double G_CONNECT_THRESHOLD = 3.7;
	private static final int DISTANCE_THRESHOLD = 0;
	private static final double GMM_CONNECT_THRESHOLD = 2;

	public ICGTClustering()
	{
		_nodesLeaf = new ArrayList<ICGTNode>();
		_nodeRoot = null;
	}

	//����һ���µ�RDD�������������ࣨ�Ȱ�����ת��Ϊgmm,Ȼ���ٰ�ÿ����˹ģ�Ͳ����˹���ģ������ٽ������ĸ��£�
	public ICGTClustering run(JavaRDD<Vector> samples) throws Exception
	{
		if(samples.count() == 0)
		{
			return this;
		}

		//�õ�GMMģ��
		int k = ensureClusterNum(samples.count());
		ICGTGaussianMixtureModel icgtGMM = MathUtil.getGMMByEM(samples, k);

		insertGMMToTree(icgtGMM);
		return this;
	}

	//���µõ���GMM����ICGT���в��������ĸ���
	private void insertGMMToTree(ICGTGaussianMixtureModel icgtGMM) throws Exception
	{
		//�Ƚ�ÿ����˹ģ����Ҷ�ӽ��ľ��룬���������ֵ�����Ƚ���㣬������뵽��С�ľ����Ҷ�ӽ����
		for(int i = 0; i < icgtGMM.numOfGaussians() ; i++)
		{
			System.out.println("numOfGaussians:" + i + " / "+ icgtGMM.numOfGaussians() );

			//Ѱ�����Ҷ�ӽڵ�
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

			//˵�������ĵ�һ�����
			if(minDistance == Double.MAX_VALUE)
			{
				_nodeRoot = new ICGTNode();
				_nodeRoot.initialize(ICGTNode.NODE_TYPE.ROOT);

				ICGTNode leafNew = new ICGTNode();
				leafNew.initialize(ICGTNode.NODE_TYPE.LEAF);

				//��˹ģ�Ͷ�Ӧ��������ID
				leafNew.setGMM(icgtGMM);

				_nodesLeaf.add(leafNew);
				_nodeRoot.addChild(leafNew);

				_nodeRoot.updateWeightOfChildren();
				_nodeRoot.mergeGuassians();

				//updateMuSigma(newLeaf);//���������ľ�ֵ��Э�������
				//modify1(root);
			}
			//С����ֵ������Ϣ����
			else if(minDistance < DISTANCE_THRESHOLD)
			{
				 leafNearest.insertGaussian(icgtGMM.gaussian(i), icgtGMM.numOfSamples(i));

				 leafNearest.getNodeFather().updateWeightOfChildren();
				 //updateMuSigma(minLeaf);//���������ľ�ֵ��Э�������
			}
			//����һ���µ�Ҷ�ӽ��
			else
			{
				System.out.println("����Ҷ�ӽڵ㿪ʼ:" );
				ICGTNode leafNew = new ICGTNode();
				leafNew.initialize(ICGTNode.NODE_TYPE.LEAF);

				leafNew.setGMM(icgtGMM);
				leafNew.addData(icgtGMM.gaussian(0).mu());

				leafNearest.getNodeFather().addChild(leafNew);
				_nodeRoot = leafNew.getNodeFather().update();
				_nodesLeaf.add(leafNew);
				System.out.println("����Ҷ�ӽڵ����:" );
				//updateMuSigma(newLeaf);//���������ľ�ֵ��Э�������
			}
			System.out.println("_nodesLeaf.size():" + i + " / "+ _nodesLeaf.size());
		}
	}


	//����item����Ŀ��ȷ��k
	private int ensureClusterNum(long num)
	{
		return (int)(num);
	}


	/****************************************************************************
	 BFSGetBestCluster:������ڵ������о��������IQ\EQֵ��С�ľ������
	 ���룺
	 @root:���ڵ�
	 @bestCluster:�����Ѿ�����������
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
	 countRation1:����ĳ�־�������IQ\EQ
	 ���룺
	 @cluster:��ž�����������
	 ****************************************************************************/
	private double clusteringQuality(LinkedList<ICGTNode> clusters)
	{
		double IQ = 0, EQ = 0;
		Iterator<ICGTNode> iIt = clusters.iterator();
		while(iIt.hasNext())
		{
			System.out.println("clusteringQuality");
			//���ھ���
			ICGTNode iNode = iIt.next();
			double temp = calculateIQ(iNode);
			IQ += temp;

			//�ؼ����
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
	 countIQ:����ĳ�־�������IQֵ
	 ���룺
	 @node:����ýڵ������и�˹�ɷֵ�IQ
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
	 countEQ:���������ڵ��EQ
	 ���룺
	 @node1:�ڵ�1
	 @node2:�ڵ�2
	 ****************************************************************************/
	private double calculateEQ(ICGTNode nodeA, ICGTNode nodB)
	{
		System.out.println("calculateEQ");
		return MathUtil.GQFDistance(nodeA.getGMM(), nodB.getGMM());
	}

	//ͨ���޸�showResults�����޸ľ�����Ժ���ʾ��ʽ
	public void showResults(JavaRDD<Vector> samples)
	{
		List<Integer> listLable = _nodeRoot.predict(samples);
		if(listLable == null){
			listLable = new ArrayList<>();
		}
		int i = 0;
		for(Integer label: listLable)
		{
			System.out.println("����----->" + label);
		};
		/*
		for(int i = 0; i < _nodesLeaf.size(); ++i)
		{
			System.out.println("��"+i+"��Ҷ�ӵľ�ֵΪ"+_nodesLeaf..gaussians()[0].mu());
			System.out.println("��"+i+"��Ҷ�ӵ�Э�������Ϊ"+_nodesLeaf.get(i).gmm.gaussians()[0].sigma());
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
