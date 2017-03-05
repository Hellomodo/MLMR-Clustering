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

	private NODE_TYPE _nodeType;    //判断给节点是否为叶子结点

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

	//更新节点参数，当新生成一个节点时，调用此方法对相关父辈节点进行参数更新
	public ICGTNode update() throws Exception {

		this.mergeGuassians();
		this.nodeSplit();

		if (this._nodeType == NODE_TYPE.ROOT) {
			return this;
		} else {
			return _nodeFather.update();
		}
	}

	//将某一非叶子节点下所有的高斯混合模型中的高斯成分合并成为该节点的高斯成分
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
		while (itNode != null)					  //保存各高斯成分参数
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


	//分裂相应的
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

		for (int i = 0; i < num; ++i)                           //构图
		{
			for (int j = i; j < num; ++j) {

				if (index[i].isLeaf() == false)                          //叶子层利用欧式距离公式计算距离
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
				} else                                        //非叶子层利用GQFD公式计算距离
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
		for (int i = 0; i < num; ++i)                    //计算连通图的个数
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

		if (mapAmount == num || mapAmount == 1) {//当连通图的个数为1或者连通图的个数与节点个数相同时，不需要分裂
			return false;
		}

		Arrays.fill(isVisit, false);

		if (_nodeType == NODE_TYPE.ROOT)                //需要分裂且要分裂的节点为根节点时
		{
			ICGTNode newNode = new ICGTNode();
			_nodeType = NODE_TYPE.OTHER;
			newNode.initialize(NODE_TYPE.ROOT);
			newNode.addChild(this);
		}

		//此节点分支从当前树中分离
		this.nodeSeperate();

		//分裂节点，在同一个连通图内的节点分到同一个节点下
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
		//updateMuSigma(nodeFather);//更新树结点的均值和协方差矩阵
		return true;
	}

/*
	//更新树结点的均值和方差
	private void updateMuAndSigma()
	{
		int dimension = this.getGMM().dimension();
		if(_nodeType == NODE_TYPE.LEAF)
		{
			//如果是叶子结点，直接将高斯模型的均值和方差作为结点的均值和方差
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
			//如果不是叶子结点，就利用论文中的均值和方差的计算公式计算新的均值和方差
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
