package com.edu.bit.cs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class ICGTNode {

	public static enum NODE_TYPE {ROOT, LEAF, OTHER}

	private static final double ZERO = 0.000001;
	private static final double G_CONNECT_THRESHOLD = 3.7;
	private static final int DISTANCE_THRESHOLD = 0;
	private static final double GMM_CONNECT_THRESHOLD = 2;

	private NODE_TYPE _nodeType;    //�жϸ��ڵ��Ƿ�ΪҶ�ӽ��

	private GaussianMixtureModel _gmm;

	private ICGTNode _nodeFather = null;
	private ICGTNode _nodeChild = null;
	private ICGTNode _nodeBrotherPre = null;
	private ICGTNode _nodeBrotherNext = null;

	private LinkedList<Sample> _samples;

	public ICGTNode()
	{
		_gmm = null;
		_samples = new LinkedList<Sample>();
	}

	public ICGTNode(GaussianMixtureModel gmm) {
		_gmm = gmm;
	}

	public void initialize(NODE_TYPE nodeType) {
		_nodeType = nodeType;
		if (_nodeType == NODE_TYPE.LEAF) {
			_gmm = null;
		}
		_nodeFather = null;
		_nodeChild = null;
		_nodeBrotherPre = null;
		_nodeBrotherNext = null;
	}

	public boolean isLeaf() {
		return _nodeType == NODE_TYPE.LEAF;
	}

	public long numOfSamples() {
		return _gmm.numOfSamples();
	}

	public int numOfChild() {
		ICGTNode nodeIt = _nodeChild;
		int count = 0;
		while (nodeIt != null) {
			count++;
			nodeIt = nodeIt.getNodeBrotherNext();
		}
		return count;
	}

	//���½ڵ��������������һ���ڵ�ʱ�����ô˷�������ظ����ڵ���в�������
	public ICGTNode update() throws Exception {

		this.mergeGuassians();
		this.nodeSplit();

		if (this._nodeType == NODE_TYPE.ROOT) {
			return this;
		} else {
			return _nodeFather.update();
		}
	}

	//��ĳһ��Ҷ�ӽڵ������еĸ�˹���ģ���еĸ�˹�ɷֺϲ���Ϊ�ýڵ�ĸ�˹�ɷ�
	public void mergeGuassians() throws Exception
	{
		if (_nodeChild == null)
			return;

		int numOfGaussiansSum = 0;
		ICGTNode itNode = _nodeChild;
		while (itNode != null)
		{
			numOfGaussiansSum += itNode.getGMM().numOfGaussians();
			itNode = itNode.getNodeBrotherNext();
		}
		MultivariateGaussian[] gaussians = new MultivariateGaussian[numOfGaussiansSum];

		itNode = _nodeChild;
		int count = 0;
		while (itNode != null)					  //�������˹�ɷֲ���
		{
			int numOfGaussians = itNode.getGMM().numOfGaussians();
			for (int i = 0; i < numOfGaussians; ++i)
			{
				gaussians[count++] = itNode.getGMM().gaussian(i);
			}
			itNode = itNode._nodeBrotherNext;
		}
		_gmm = new GaussianMixtureModel(gaussians);
	}


	public void nodeSeperate() {
		if (_nodeType == NODE_TYPE.ROOT)
			return;

		if (_nodeBrotherPre == null) {
			this.getNodeFather().setNodeChild(this._nodeBrotherNext);
			if (_nodeBrotherNext != null) {
				_nodeBrotherNext.setNodeBrotherPre(null);
			}
		} else if (_nodeBrotherNext == null) {
			_nodeBrotherPre.setNodeBrotherNext(null);
		} else {
			_nodeBrotherPre.setNodeBrotherNext(_nodeBrotherNext);
			_nodeBrotherNext.setNodeBrotherPre(_nodeBrotherPre);
		}
	}


	//������Ӧ��
	public boolean nodeSplit() throws Exception {
		int num = numOfChild();
		if (num == 1)
			return false;

		boolean[][] matConn = new boolean[num][num];
		boolean[] isVisit = new boolean[num];

		ICGTNode nodeIt = _nodeChild;
		ICGTNode[] index = new ICGTNode[num];
		for (int i = 0; i < num; ++i) {
			index[i] = nodeIt;
			nodeIt = nodeIt.getNodeBrotherNext();
		}

		for (int i = 0; i < num; ++i)                           //��ͼ
		{
			for (int j = i; j < num; ++j) {

				if (index[i].isLeaf() == false)                          //Ҷ�Ӳ�����ŷʽ���빫ʽ�������
				{
					double temp;
					if (i != j) {
						temp = MathUtil.GQFDistance(index[i].getGMM(), index[j].getGMM());
					} else {
						temp = ZERO;
					}

					if (temp < GMM_CONNECT_THRESHOLD) {
						matConn[i][j] = true;
						matConn[j][i] = true;
					}
				} else                                        //��Ҷ�Ӳ�����GQFD��ʽ�������
				{
					double temp;
					if (i != j) {
						temp = MathUtil.eulcideanDistance(index[i].getGMM().gaussian(0), index[j].getGMM().gaussian(0));
					} else {
						temp = ZERO;
					}

					if (temp < G_CONNECT_THRESHOLD) {
						matConn[i][j] = true;
						matConn[j][i] = true;
					}
				}
			}
		}

		MathUtil.warshall(matConn);

		int mapAmount = 0;
		for (int i = 0; i < num; ++i)                    //������ͨͼ�ĸ���
		{
			if (!isVisit[i]) {
				mapAmount++;
				for (int j = 0; j < num; ++j) {
					if (matConn[i][j]) {
						isVisit[j] = true;
					}
				}
			}
		}

		if (mapAmount == num || mapAmount == 1) {//����ͨͼ�ĸ���Ϊ1������ͨͼ�ĸ�����ڵ������ͬʱ������Ҫ����
			return false;
		}

		Arrays.fill(isVisit, false);

		if (_nodeType == NODE_TYPE.ROOT)                //��Ҫ������Ҫ���ѵĽڵ�Ϊ���ڵ�ʱ
		{
			ICGTNode newNode = new ICGTNode();
			_nodeType = NODE_TYPE.OTHER;
			newNode.initialize(NODE_TYPE.ROOT);
			newNode.addChild(this);
		}

		//�˽ڵ��֧�ӵ�ǰ���з���
		this.nodeSeperate();

		//���ѽڵ㣬��ͬһ����ͨͼ�ڵĽڵ�ֵ�ͬһ���ڵ���
		for (int i = 0; i < num; ++i) {
			if (!isVisit[i]) {
				ICGTNode newNode;
				newNode = new ICGTNode();
				newNode.initialize(NODE_TYPE.OTHER);
				_nodeFather.addChild(newNode);
				for (int j = 0; j < num; ++j) {
					if (true == matConn[i][j]) {
						isVisit[j] = true;
						index[j].nodeSeperate();
						newNode.addChild(index[j]);
					}

				}
				newNode.mergeGuassians();
			}
		}
		_nodeFather.update();
		//updateMuSigma(nodeFather);//���������ľ�ֵ��Э�������
		return true;
	}

/*
	//���������ľ�ֵ�ͷ���
	private void updateMuAndSigma()
	{
		int dimension = this.getGMM().dimension();
		if(_nodeType == NODE_TYPE.LEAF)
		{
			//�����Ҷ�ӽ�㣬ֱ�ӽ���˹ģ�͵ľ�ֵ�ͷ�����Ϊ���ľ�ֵ�ͷ���
			double[] array = new double[dimension*dimension];
			for(int i = 0 ;i < node.dimension ;++i)
			{
				array[i] = node.gmm.gaussians()[0].mu().apply(i);
			}
			node.mu = new DenseVector(array);
			for(int i = 0; i < node.dimension; ++i)
			{
				for(int j = 0; j < node.dimension; ++j)
				{
					array[i*node.dimension+j] = node.gmm.gaussians()[0].sigma().apply(i, j);
				}
			}
			node.sigma = new DenseMatrix(node.dimension, node.dimension, array);
			return;
		}
		else
		{
			//�������Ҷ�ӽ�㣬�����������еľ�ֵ�ͷ���ļ��㹫ʽ�����µľ�ֵ�ͷ���
			GMMTreeNode iter = node.son;
			double[] arrMu = new double[node.dimension];
			double[] arrSigma = new double[node.dimension*node.dimension];
			long sum = 0;
			while(iter != null)
			{
				arrMu = calculateMu(sum,arrMu,iter);
				arrSigma = calculateSigma(sum,arrSigma,iter);
				sum += iter.dataNum;
				iter = iter.next_bro;
			}
			node.mu = new DenseVector(arrMu);
			node.sigma = new DenseMatrix(node.dimension, node.dimension, arrSigma);
		}
		updateMuSigma(node.father);
	}
*/

	public void addChild(ICGTNode node) {
		node.setNodeFather(this);
		ICGTNode tmp = _nodeChild;
		_nodeChild = node;
		node.setNodeBrotherNext(tmp);
		if (null != tmp) {
			tmp.setNodeBrotherPre(node);
		}
	}

	public ArrayList<ICGTNode> getNodesChildren()
	{
		ArrayList<ICGTNode> children = new ArrayList<ICGTNode>();
		ICGTNode nodeIt = _nodeChild;
		while(nodeIt != null)
		{
			children.add(nodeIt);
			nodeIt = nodeIt.getNodeBrotherNext();
		}
		return children;
	}

	public void addSample(Sample sample) {
		_samples.offer(sample);
	}

	public LinkedList<Sample> getSample() {
		return _samples;
	}

	public void setGMM(GaussianMixtureModel gmm)
	{
		_gmm = gmm;
	};

	public GaussianMixtureModel getGMM()
	{
		return _gmm;
	};

	public void setNodeFather(ICGTNode nodeFather)
	{
		_nodeFather = nodeFather;
	}

	public ICGTNode getNodeFather()
	{
		return _nodeFather;
	}

	public void setNodeChild(ICGTNode nodeChild)
	{
		_nodeChild = nodeChild;
	}

	public ICGTNode getNodeChild()
	{
		return _nodeChild;
	}

	public void setNodeBrotherPre(ICGTNode nodeBrotherPre)
	{
		_nodeBrotherPre = nodeBrotherPre;
	}

	public ICGTNode getNodeBrotherPre()
	{
		return _nodeBrotherPre;
	}

	public void setNodeBrotherNext(ICGTNode nodeBrotherNext)
	{
		_nodeBrotherNext = nodeBrotherNext;
	}

	public ICGTNode getNodeBrotherNext()
	{
		return _nodeBrotherNext;
	}

}
