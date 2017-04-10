package com.edu.bit.cs;

import java.util.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import javax.swing.JFrame;


import org.apache.spark.mllib.linalg.Vector;

public class ICGTClustering
{
	//���ĸ��ڵ�
	private ICGTNode _nodeRoot;

	private LinkedList<Sample> _queueSamples;
	//����ȫ��Ҷ�ӽ��
	private LinkedList<ICGTNode> _nodesLeaf;

	//һЩ�������õ��ĳ���ֵ
	private static final double ZERO = 0.000001;
	private static final double DISTANCE_THRESHOLD =0;

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
		_nodeRoot.threshold(150);
		//_nodeRoot = _nodeRoot.update();


		while(true)
		{
			ArrayList<ICGTNode> clusters = _nodeRoot.getNodesChildren();
			double minDistance = Double.MAX_VALUE;
			ICGTNode nodeA = null, nodeB = null;
			for(int i = 0; i < clusters.size(); i++)
			{
				for(int j = 0; j < clusters.size(); j++)
				{

					System.out.println(clusters.size() + ":��" + i + "," + j + ")");

					double distance = MathUtil.GQFDistance(clusters.get(i).getGMM(), clusters.get(j).getGMM());
					if(distance < minDistance && i != j)
					{
						minDistance = distance;
						nodeA = clusters.get(i);
						nodeB = clusters.get(j);
					}
					//System.out.println(i + ":" + j + "-->" + MathUtil.KLDivergence(clusters.get(i).getGMM(), clusters.get(j).getGMM()));
				}
			}

			nodeA.seperateFromParents();
			nodeB.seperateFromParents();
			ICGTNode node = new ICGTNode(ICGTNode.NODE_TYPE.OTHER);
			node.addChild(nodeA);
			node.addChild(nodeB);
			node.mergeGuassians();
			_nodeRoot.addChild(node);
			_nodeRoot.mergeGuassians();

			if(_nodeRoot.numOfChild() == 4)
				break;
		}

		ArrayList<ICGTNode> clusters = _nodeRoot.getNodesChildren();
		for(int i = 0; i < clusters.size(); i++)
		{
			for(int j = 0; j < clusters.size(); j++)
			{
				System.out.println(i + ":" + j + "-->" + MathUtil.GQFDistance(clusters.get(i).getGMM(), clusters.get(j).getGMM()));
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


	//����һ���µ�RDD�������������ࣨ�Ȱ�����ת��Ϊgmm,Ȼ���ٰ�ÿ����˹ģ�Ͳ����˹���ģ������ٽ������ĸ��£�
	public ICGTClustering run(Iterator<Vector> samples)
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

            System.out.println("Ѱ�������Ҷ�ӽڵ�: "+ count + "/" );
			//Ѱ�����Ҷ�ӽڵ�
			double minDistance = Double.MAX_VALUE;
			ICGTNode leafNearest = null;
			Iterator<ICGTNode> itNode = _nodesLeaf.iterator();
			while (itNode.hasNext())
			{
				ICGTNode nodeTmp = itNode.next();
				double distance = MathUtil.KLDivergence( gmmNew, nodeTmp.getGMM());
				if(distance < minDistance)
				{
					minDistance = distance;
					leafNearest = nodeTmp;
				}
			}
            System.out.println("�ҵ������Ҷ�ӽڵ�");
			//˵�������ĵ�һ�����
			if(minDistance == Double.MAX_VALUE)
			{
				_nodeRoot = new ICGTNode(ICGTNode.NODE_TYPE.ROOT);

				ICGTNode leafNew = new ICGTNode(ICGTNode.NODE_TYPE.LEAF);

				//��˹ģ�Ͷ�Ӧ��������ID
				leafNew.setGMM(gmmNew);
				leafNew.addSample(sample);
				_nodesLeaf.add(leafNew);
				_nodeRoot.addChild(leafNew);
				_nodeRoot.mergeGuassians();

				//updateMuSigma(newLeaf);//���������ľ�ֵ��Э�������
				//modify1(root);
			}
			//С����ֵ������Ϣ����
			else if(minDistance < DISTANCE_THRESHOLD)
			{
				leafNearest.setGMM(new GaussianMixtureModel(MathUtil.mergeGaussians(leafNearest.getGMM().gaussian(0),gmmNew.gaussian(0))));
				leafNearest.addSample(sample);
				_nodeRoot.mergeGuassians();
				_nodeRoot = leafNearest.getNodeFather().update();
				//updateMuSigma(minLeaf);//���������ľ�ֵ��Э�������
			}
			//����һ���µ�Ҷ�ӽ��
			else
			{
				ICGTNode leafNew = new ICGTNode(ICGTNode.NODE_TYPE.LEAF);

				leafNew.setGMM(gmmNew);
				leafNew.addSample(sample);

				leafNearest.getNodeFather().addChild(leafNew);
				_nodeRoot = leafNew.getNodeFather().update();
 				_nodesLeaf.add(leafNew);
				//updateMuSigma(newLeaf);//���������ľ�ֵ��Э�������
			}
            System.out.println("����Ҷ�ӽڵ�");
		}
		return this;
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
			//���ھ���
			ICGTNode iNode = iIt.next();
			double temp = calculateIQ(iNode);
			double tmp = IQ + temp;
            IQ = tmp;
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
	 countEQ:���������ڵ��EQ
	 ���룺
	 @node1:�ڵ�1
	 @node2:�ڵ�2
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
