package com.edu.bit.cs;


import java.io.Serializable;
import java.util.*;

public class ICGTNode  implements Serializable {

	public static enum NODE_TYPE {ROOT, LEAF, OTHER}

	private double ZERO = 0.000001;

	private double _thresholdGMM = 0;
	private double _thresholdMG = 0;

	private int _numOfChildren;
	private NODE_TYPE _nodeType;    //判断给节点是否为叶子结点

	private GaussianMixtureModel _gmm;

	private boolean _isChanged;
	private boolean _isClosure;

	private ICGTNode _nodeFather;
	private ICGTNode _nodeChild;
	private ICGTNode _nodeBrotherPre;
	private ICGTNode _nodeBrotherNext;


	public void threshold(double threshold)
	{
		_thresholdGMM = threshold;
	}

	public double threshold()
	{
		return _thresholdGMM;
	}

	private LinkedList<Sample> _samples;

	public ICGTNode(NODE_TYPE nodeType)
	{
		initialize(nodeType);
	}

	public void initialize(NODE_TYPE nodeType) {
		_nodeType = nodeType;
		_nodeFather = null;
		_nodeChild = null;
		_nodeBrotherPre = null;
		_nodeBrotherNext = null;
		_isChanged = true;
		_numOfChildren = 0;
		if (_nodeType == NODE_TYPE.LEAF) {
			_samples = new LinkedList<Sample>();
		}
		else{
			_isClosure = false;
		}
	}

	//更新节点参数，当新生成一个节点时，调用此方法对相关父辈节点进行参数更新
	public ICGTNode update() {

		this.mergeGuassians();
		this.nodeSplit();
		//this.nodeBiSplit();
		_isChanged = true;
		if (this._nodeType == NODE_TYPE.ROOT) {
			return this;
		} else {
			return _nodeFather.update();
		}

	}

	//将某一非叶子节点下所有的高斯混合模型中的高斯成分合并成为该节点的高斯成分
	public void mergeGuassians()
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

	public void seperateFromParents() {
		if (_nodeType == NODE_TYPE.ROOT)
			return;

		if (_nodeBrotherPre == null) {
			_nodeFather.setNodeChild(this._nodeBrotherNext);
			if (_nodeBrotherNext != null) {
				_nodeBrotherNext.setNodeBrotherPre(null);
			}
		} else if (_nodeBrotherNext == null) {
			_nodeBrotherPre.setNodeBrotherNext(null);
		} else {
			_nodeBrotherPre.setNodeBrotherNext(_nodeBrotherNext);
			_nodeBrotherNext.setNodeBrotherPre(_nodeBrotherPre);
		}
		_nodeFather._numOfChildren --;
	}

	public boolean nodeBiSplit() {
		int num = _numOfChildren;
		if (num < 200)
			return false;

		ICGTNode newNode = new ICGTNode(_nodeType);
		if(_nodeType != NODE_TYPE.ROOT)
		{
			_nodeFather.addChild(newNode);
		}

		//此节点分支从当前树中分离
		this.seperateFromParents();

		ICGTNode[] nodes = new ICGTNode[2];
		nodes[0] = new ICGTNode(NODE_TYPE.OTHER);
		nodes[1] = new ICGTNode(NODE_TYPE.OTHER);
		//分裂节点
		ICGTNode nodeIt = _nodeChild;
		for (int i = 0; i < num; ++i) {
			ICGTNode tmp = nodeIt.getNodeBrotherNext();
			nodeIt.seperateFromParents();
			nodes[i/(num/2)].addChild(nodeIt);
			nodeIt = tmp;
		}
		nodes[0].mergeGuassians();
		nodes[0].isClosure(_isClosure);
		nodes[1].mergeGuassians();
		nodes[1].isClosure(_isClosure);
		newNode.addChild(nodes[0]);
		newNode.addChild(nodes[1]);
		newNode.mergeGuassians();
		return true;
	}

	//分裂相应的
	public boolean nodeSplit() {
		int num = _numOfChildren;
		if (num == 1)
			return false;

		ArrayList<Integer> indexChanged = new ArrayList<Integer>();

		ICGTNode[] children = new ICGTNode[num];
		UnionFindSet ufs = new UnionFindSet(num);

		ICGTNode nodeIt = _nodeChild;
		int firstConstant = -1;
		for (int i = 0; i < num; ++i) {
			children[i] = nodeIt;
			if(nodeIt.isChanged()) {
				indexChanged.add(i);
			}else if(_isClosure ) {
				if(firstConstant == -1)
					firstConstant = i;
				else
					ufs.union(i, firstConstant);
			}
			nodeIt = nodeIt.getNodeBrotherNext();
		}

		// 遍历，完整并查集,找出最大闭包
		for(int j = 0; j < indexChanged.size() && indexChanged.get(j) != -1; j ++)
		{
			GaussianMixtureModel gmmChanged = children[indexChanged.get(j)].getGMM();
			MultivariateGaussian gaussianChanged = children[indexChanged.get(j)].getGMM().gaussian(0);
			for (int i = 0; i < num; ++i) {                          //构图
				double temp;
				if (children[i].isLeaf() == false) {                      //非叶子层利用GQFD公式计算距离
					if (i != indexChanged.get(j)) {
						temp = MathUtil.KLDivergence(children[i].getGMM(), gmmChanged);
					} else {
						continue;
					}

					if (temp < _thresholdGMM) {
						ufs.union(i,indexChanged.get(j));
					}
				}
				else{                                        //叶子层利用欧式距离公式计算距离

					if (i != indexChanged.get(j)) {
						temp = MathUtil.eulcideanDistance(children[i].getGMM().gaussian(0),gaussianChanged);
					} else {
						continue;
					}

					if (temp < _thresholdMG) {
						ufs.union(i,indexChanged.get(j));
					}
				}
			}
		}

		//System.out.println("闭包数目"+ufs.count() );
		//当连通图的个数为1或者连通图的个数与节点个数相同时，不需要分裂
		if (ufs.count() == num) {
			_isClosure = false;
			return false;
		}
		else if(ufs.count() == 1){
			_isClosure = true;
			return false;
		}

		//需要分裂且要分裂的节点为根节点时
		if (_nodeType == NODE_TYPE.ROOT)
		{
			ICGTNode newNode = new ICGTNode(NODE_TYPE.ROOT);
			_nodeType = NODE_TYPE.OTHER;
			newNode.addChild(this);
		}

		//此节点分支从当前树中分离
		this.seperateFromParents();

		Map<Integer,ICGTNode> mapNode = new HashMap<>();

		//新增结点，并包含各自闭包子节点
		for (int i = 0; i < num; ++i) {
			children[i].seperateFromParents();
			if( mapNode.containsKey(ufs.find(i)) ){
				mapNode.get(ufs.find(i)).addChild(children[i]);
			}
			else {
				ICGTNode node =  new ICGTNode(NODE_TYPE.OTHER);
				node.addChild(children[i]);
				mapNode.put(ufs.find(i),node);

			}

		}

		Iterator<ICGTNode> itNode = mapNode.values().iterator();
		while (itNode.hasNext()) {
			ICGTNode node = itNode.next();
			node.isClosure(true);
			_nodeFather.addChild(node);
			node.mergeGuassians();
		}

		return true;
	}

	public void addChild(ICGTNode node) {
		_isChanged = true;
		node.setNodeFather(this);
		ICGTNode tmp = _nodeChild;
		_nodeChild = node;
		node.setNodeBrotherNext(tmp);
		if (null != tmp) {
			tmp.setNodeBrotherPre(node);
		}
		node.threshold(_thresholdGMM);
		_numOfChildren ++;
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


	public void isChanged(boolean isChanged)
	{
		_isChanged = isChanged;
	}


	public boolean isChanged()
	{
		if(_isChanged) {
			_isChanged = false;
			return true;
		}
		else {
			return false;
		}
	}

	public boolean isLeaf() {
		return _nodeType == NODE_TYPE.LEAF;
	}

	public void isClosure(boolean isClosure)
	{
		_isClosure = isClosure;
	}

	public void addSample(Sample sample) {
		_samples.offer(sample);
	}

	public LinkedList<Sample> getSample() {
		return _samples;
	}

	public void setGMM(GaussianMixtureModel gmm)
	{
		_isChanged = true;
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
