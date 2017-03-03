package com.edu.bit.cs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class ICGTNode {

	public static enum NODE_TYPE {ROOT, LEAF, OTHER}

	private static final double ZERO = 0.000001;
	private static final double G_CONNECT_THRESHOLD = 5;
	private static final int DISTANCE_THRESHOLD = 5;
	private static final double GMM_CONNECT_THRESHOLD = 5;

	private NODE_TYPE _nodeType;    //判断给节点是否为叶子结点

	private ICGTGaussianMixtureModel _icgtGMM;

	private ICGTNode _nodeFather = null;
	private ICGTNode _nodeChild = null;
	private ICGTNode _nodeBrotherPre = null;
	private ICGTNode _nodeBrotherNext = null;

	private LinkedList<Vector> _dataList;

	private double _weight = 1;

	public ICGTNode()
	{
		_icgtGMM = null;
		_dataList = new LinkedList<Vector>();
	}

	public ICGTNode(ICGTGaussianMixtureModel icgtGMM) {
		_icgtGMM = icgtGMM;
	}

	public void initialize(NODE_TYPE nodeType) {
		_nodeType = nodeType;
		if (_nodeType == NODE_TYPE.LEAF) {
			_icgtGMM = null;
		}
		_nodeFather = null;
		_nodeChild = null;
		_nodeBrotherPre = null;
		_nodeBrotherNext = null;
	}

	public void addData(Vector data) {
		_dataList.offer(data);
	}

	public LinkedList<Vector> getData() {
		return _dataList;
	}

	public boolean isLeaf() {
		return _nodeType == NODE_TYPE.LEAF;
	}

	public long numOfSamples() {
		return _icgtGMM.numOfSamples();
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

	public void addChild(ICGTNode node) {
		node.setNodeFather(this);
		ICGTNode tmp = _nodeChild;
		_nodeChild = node;
		node.setNodeBrotherNext(tmp);
		if (null != tmp) {
			tmp.setNodeBrotherPre(node);
		}
	}

	//更新节点参数，当新生成一个节点时，调用此方法对相关父辈节点进行参数更新
	public ICGTNode update() throws Exception {
		System.out.println("updateWeightOfChildren:" );
		updateWeightOfChildren();
		System.out.println("mergeGuassians:" );
		mergeGuassians();
		System.out.println("nodeSplit:" );
		this.nodeSplit();
		System.out.println("return:" );
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
		ICGTGaussianMixtureModel rslt = _nodeChild.getGMM();

		ICGTNode nodeIt = _nodeChild.getNodeBrotherNext();
		while (nodeIt != null)
		{
			rslt = MathUtil.mergeGMM(rslt, nodeIt.getGMM());
			nodeIt = nodeIt.getNodeBrotherNext();
		}
		_icgtGMM = rslt;
	}

	//修改结点node的权重信息
	public void updateWeightOfChildren() {
		if (_nodeType == NODE_TYPE.LEAF)
			return;
		int numOfSamples = 0;
		ICGTNode nodeIt = _nodeChild;
		while (nodeIt != null) {
			numOfSamples += nodeIt.numOfSamples();
			nodeIt = nodeIt.getNodeBrotherNext();
		}
		nodeIt = _nodeChild;
		while (nodeIt != null) {
			nodeIt.setWeight((double) (nodeIt.numOfSamples()) / numOfSamples);
			nodeIt = nodeIt.getNodeBrotherNext();
		}
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
				newNode.updateWeightOfChildren();
				newNode.mergeGuassians();
			}
		}
		_nodeFather.update();
		//updateMuSigma(nodeFather);//更新树结点的均值和协方差矩阵
		return true;
	}

	//将单个高斯模型插入树的叶子结点里
	public void insertGaussian(MultivariateGaussian gaussian, long numOfSamplesArg) throws Exception {
		int dimension = gaussian.mu().size();
		long numOfSamplesThis = this.numOfSamples();
		long numOfSamples = numOfSamplesArg + numOfSamplesThis;
		Vector vecMuThis = _icgtGMM.gaussian(0).mu();
		Matrix matSigmaThis = _icgtGMM.gaussian(0).sigma();
		Vector vecMuArg = gaussian.mu();
		Matrix matSigmaArg = gaussian.sigma();
		double[] muRslt = new double[dimension];
		double[] sigmaRslt = new double[dimension];


		//获得新的均值和协方差矩阵
		//u = ((node).u *(node).num + (gau).u * (gau).num)/((node).num+(gau).num)
		for (int i = 0; i < dimension; ++i) {

			muRslt[i] = vecMuThis.apply(i) * numOfSamplesThis / (numOfSamplesThis + numOfSamplesArg) + vecMuArg.apply(i) * numOfSamplesArg / (numOfSamplesThis + numOfSamplesArg);

			double tmp = vecMuArg.apply(i) - vecMuThis.apply(i);

			sigmaRslt[i] = (numOfSamplesThis - 1) * matSigmaThis.apply(i, i) / (numOfSamples - 1)
					+ (numOfSamplesThis - 1) * matSigmaArg.apply(i, i) / (numOfSamples - 1)
					+ numOfSamplesThis * numOfSamplesArg * tmp * tmp / (numOfSamples * (numOfSamples - 1));
		}

		_icgtGMM = new ICGTGaussianMixtureModel(new MultivariateGaussian(new DenseVector(muRslt), DenseMatrix.diag(Vectors.dense(sigmaRslt))), numOfSamplesThis + numOfSamplesArg);
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


	public List<Integer> predict(JavaRDD<Vector> samples) {
		return _icgtGMM.predict(samples);
	}

	public ArrayList<ICGTNode> getChildren()
	{
		ArrayList<ICGTNode> children = new ArrayList<ICGTNode>();
		ICGTNode nodeIt = _nodeChild;
		while(nodeIt != null)
		{
			children.add(nodeIt);
			nodeIt.getNodeBrotherNext();
		}
		return children;
	}

	public void setWeight(double weight)
	{
		_weight = weight;
	}

	public double getWeight()
	{
		return _weight;
	}

	public void setGMM(ICGTGaussianMixtureModel icgtGMM)
	{
		_icgtGMM = icgtGMM;
	};

	public ICGTGaussianMixtureModel getGMM()
	{
		return _icgtGMM;
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
