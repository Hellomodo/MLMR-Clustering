package com.edu.bit.cs;

import java.util.*;

public class MathUtil
{
	//一些运算里用到的常量值
	private static final double ZERO = 0.000001;

	//合并两个高斯模型
	public static MultivariateGaussian mergeGaussians(MultivariateGaussian gaussianA ,MultivariateGaussian gaussianB)
	{
		int dimension = gaussianA.dimension();
		long numOfSamplesA = gaussianA.numOfSamples();
		long numOfSamplesB = gaussianB.numOfSamples();
		long numOfSamples = numOfSamplesA + numOfSamplesB;
		double[] meanA = gaussianA.mean();
		double[] covA = gaussianA.cov();
		double[] meanB = gaussianB.mean();
		double[] covB = gaussianB.cov();
		double[] meanMerged = new double[dimension];
		double[] covMerged = new double[dimension];


		//获得新的均值和协方差矩阵
		//u = ((node).u *(node).num + (gau).u * (gau).num)/((node).num+(gau).num)
		for (int i = 0; i < dimension; ++i)
		{

			meanMerged[i] = meanA[i] * numOfSamplesA / (numOfSamplesA + numOfSamplesB) + meanB[i] * numOfSamplesB / (numOfSamplesA + numOfSamplesB);

			double tmp = meanB[i] -  meanA[i];

			covMerged[i] = (numOfSamplesA - 1) * covA[i] / (numOfSamples - 1)
					+ (numOfSamplesB - 1) * covB[i] / (numOfSamples - 1)
					+ numOfSamplesA * numOfSamplesB * tmp * tmp / (numOfSamples * (numOfSamples - 1));
		}

		return new MultivariateGaussian(meanMerged, covMerged, numOfSamples);
	}

	//KL散度距离计算公式
	public static double KLDivergence (MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
	{
		double resultAtoB = 0;
		int dimension = gaussianA.dimension();
		double[] meanA = gaussianA.mean();
		double[] covA = gaussianA.cov();
		double[] meanB = gaussianB.mean();
		double[] covB = gaussianB.cov();

		for (int i = 0; i < dimension; i++)
		{
			if (covB[i] == 0)
			{
				covB[i] = ZERO;
			}
			if (covA[i] == 0)
			{
				resultAtoB += covA[i] / covB[i]+ Math.pow((meanA[i] - meanB[i]), 2.0) / covB[i];
			}
			else
			{
				resultAtoB += covA[i] / covB[i] + Math.pow((meanA[i] - meanB[i]), 2.0) / covB[i]
						- Math.log(Math.abs(covA[i] / covB[i]));
			}

		}

		resultAtoB -= dimension;
		resultAtoB *= 0.5;
		return resultAtoB;
		//return Math.atan(resultAtoB) * 2 / Math.PI;
	}


	//计算两个高斯混合模型的距离
	public static  double GQFDistance(GaussianMixtureModel icgtGMMA,GaussianMixtureModel icgtGMMB)
	{
		int numOfGassiansA = icgtGMMA.numOfGaussians();
		int numOfGassiansB = icgtGMMB.numOfGaussians();
		int numOfGaussians = numOfGassiansA + numOfGassiansB;

		double[] weights = new double[numOfGaussians];
		System.arraycopy(icgtGMMA.weights(), 0, weights, 0, numOfGassiansA);
		//System.arraycopy(icgtGMMB.weights(), 0, weights, numOfGassiansA, numOfGassiansB);
		for(int i = 0; i < numOfGassiansB; i ++)
		{
			weights[i + numOfGassiansA] = -icgtGMMB.weights()[i];
		}

		MultivariateGaussian[] gaussians = new MultivariateGaussian[numOfGaussians];
		System.arraycopy(icgtGMMA.gaussians(), 0, gaussians, 0, numOfGassiansA);
		System.arraycopy(icgtGMMB.gaussians(), 0,gaussians, numOfGassiansA, numOfGassiansB);

		double result = 0;
		for (int i = 0; i < numOfGaussians; i++)
		{
			for (int j = 0; j < numOfGaussians; j++)
			{
				double KLdistance = Math.min( MathUtil.KLDivergence(gaussians[i], gaussians[j]), MathUtil.KLDivergence(gaussians[j], gaussians[i]));
				double Aij = 1 / (KLdistance + 1);
				result += weights[i] * Aij * weights[j];
			}
		}

		result = result < 0 ? 0 : result;
		return Math.sqrt(result);
	}


	//计算两个高斯混合模型的距离
	public static  double KLDivergence(GaussianMixtureModel icgtGMMA,GaussianMixtureModel icgtGMMB)
	{
		int numOfGassiansA = icgtGMMA.numOfGaussians();
		int numOfGassiansB = icgtGMMB.numOfGaussians();

		MultivariateGaussian[] gaussianA = icgtGMMA.gaussians();
		MultivariateGaussian[] gaussianB = icgtGMMB.gaussians();

		double[] weightsA = icgtGMMA.weights();
		double[] weightsB = icgtGMMB.weights();

		double KLDiverAtoB = 0, KLDiverBtoA = 0;
		for (int i = 0; i < numOfGassiansA; i++)
		{
			double tmpAtoA = 0, tmpAtoB = 0;

			for (int j = 0; j < numOfGassiansA; j++)
			{
				double KLForGaussians = MathUtil.KLDivergence(gaussianA[i], gaussianA[j]);
				tmpAtoA += weightsA[j] * Math.exp(-KLForGaussians);
			}

			for (int j = 0; j < numOfGassiansB; j++)
			{
				double KLForGaussians = MathUtil.KLDivergence(gaussianA[i], gaussianB[j]);
				tmpAtoB += weightsB[j] * Math.exp(-KLForGaussians);
			}
			KLDiverAtoB += weightsA[i] + Math.log( tmpAtoA / tmpAtoB );
		}

		for (int i = 0; i < numOfGassiansB; i++)
		{
			double tmpBtoB = 0, tmpBtoA = 0;

			for (int j = 0; j < numOfGassiansB; j++)
			{
				double KLForGaussians = MathUtil.KLDivergence(gaussianB[i], gaussianB[j]);
				tmpBtoB += weightsB[j] * Math.exp(-KLForGaussians);
			}

			for (int j = 0; j < numOfGassiansA; j++)
			{
				double KLForGaussians = MathUtil.KLDivergence(gaussianB[i], gaussianA[j]);
				tmpBtoA += weightsA[j] * Math.exp(-KLForGaussians);
			}
			KLDiverBtoA += weightsB[i] + Math.log( tmpBtoB / tmpBtoA );
		}

		double result = Math.min(KLDiverAtoB,KLDiverBtoA);
		return (Math.atan(result) * 2 / Math.PI - 0.5) * 2;
	}




	//计算两个高斯模型的欧式距离距离
	public static double eulcideanDistance(MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
	{
		double distance = 0;
		int dimension = gaussianA.dimension();
		double[] meanA = gaussianA.mean();
		double[] meanB = gaussianB.mean();
		for (int i = 0; i < dimension ; ++i)
		{
			double temp = meanA[i] - meanB[i];
			distance += temp * temp;
		}
		distance = Math.sqrt(distance); //欧氏距离
		return distance;
	}

	public static Map<Integer,Set<Integer>> clusterDistri(int[] A){
		Map<Integer,Set<Integer>> clusterD = new HashMap<Integer,Set<Integer>>();
		int max = -1;
		for(int i = 0;i< A.length;i++){

			if(max < A[i]){
				max = A[i];
			}
		}
		for(int i = 0;i< A.length;i++){
			int temp = A[i];
			if(temp < max+1){
				if(clusterD.containsKey(temp)){
					Set<Integer> set = clusterD.get(temp);
					set.add(i+1);
					clusterD.put(temp, set);
				}else{
					Set<Integer> set = new HashSet<Integer>();
					set.add(i+1);
					clusterD.put(temp, set);
				}
			}
		}
		return clusterD;
	}

	public static double ClusEvaluate(String method,List<Sample> samples){

		int[] labels = new int[samples.size()];
		int[] pridicts = new int[samples.size()];
		for(int i = 0; i < samples.size(); i++)
		{
			labels[i] = samples.get(i).getLabel();
			pridicts[i] = samples.get(i).getPridict();
		}
		switch(method){
			case "Purity":
				return Purity(labels,pridicts);
			case "Precision":
				return Precision(labels,pridicts);
			case "Recall":
				return Recall(labels,pridicts);
			case "RI":
				return RI(labels,pridicts);
			case "F_score":
				return F_score(labels,pridicts);
			case "NMI":
				return NMI(labels,pridicts);
			case "Jaccard":
				return Jaccard(labels,pridicts);
			default:
				return -1.0;
		}

	}
	public static int[] commNum(Map<Integer,Set<Integer>> A,Map<Integer,Set<Integer>> B){
		int[] commonNo = new int[A.size()];
		int com = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = A.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			Set<Integer> setA = entryA.getValue();
			Iterator<Map.Entry<Integer,Set<Integer>>> itB = B.entrySet().iterator();
			int maxComm = -1;
			while(itB.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryB = itB.next();
				Set<Integer> setB = entryB.getValue();
				int lengthA = setA.size();
				Set<Integer> temp = new HashSet<Integer>(setA);

				temp.removeAll(setB);

				int lengthCom = lengthA - temp.size();

				if(maxComm < lengthCom){
					maxComm = lengthCom;
				}

			}

			commonNo[i] = maxComm;
			com = com + maxComm;
			i++;
		}

		return commonNo;
	}
	/*
     * 所有簇分配正确的除以总的。其中B是对比的标准标签。
     */
	public static double Purity(int[] A,int[] B){
		double value;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);
		int[] commonNo = commNum(clusterA,clusterB);
		int com = 0;
		for(int i = 0;i<commonNo.length;i++){
			com = com + commonNo[i];
		}
		value = com*1.0/A.length;

		return value;
	}
	/*
     * @param A,B
     * @return 精度
     */
	public static double Precision(int[] A,int[] B){
		double value = 0.0;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		int[] commonNo = commNum(clusterA,clusterB);//得到A中每个簇中聚类正确的数目。
		int allP = 0;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			allP = allP + combination(entryA.getValue().size(),2);
			TP = TP + combination(commonNo[i],2);
			i++;
		}

		FP = allP - TP;

		itA = clusterA.entrySet().iterator();
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();

			Iterator<Map.Entry<Integer,Set<Integer>>> itA2 = clusterA.entrySet().iterator();
			while(itA2.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryA2 = itA2.next();
				if(entryA != entryA2){
					Set<Integer> s1 = entryA.getValue();
					Set<Integer> s2 = entryA2.getValue();
					for(Integer i1 :s1){
						for(Integer i2:s2){
							if(B[i1-1] != B[i2-1]){
								TN++;
							}else{
								FN++;
							}
						}
					}

				}
			}
		}

		double P = TP*1.0/(TP + FP);
		return P;
	}
	/*
     * @param A,B
     * @return recal召回率
     */
	public static double Recall(int[] A,int[] B){
		double value = 0.0;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		int[] commonNo = commNum(clusterA,clusterB);//得到A中每个簇中聚类正确的数目。
		int allP = 0;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			allP = allP + combination(entryA.getValue().size(),2);
			TP = TP + combination(commonNo[i],2);
			i++;
		}

		FP = allP - TP;

		itA = clusterA.entrySet().iterator();
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();

			Iterator<Map.Entry<Integer,Set<Integer>>> itA2 = clusterA.entrySet().iterator();
			while(itA2.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryA2 = itA2.next();
				if(entryA != entryA2){
					Set<Integer> s1 = entryA.getValue();
					Set<Integer> s2 = entryA2.getValue();
					for(Integer i1 :s1){
						for(Integer i2:s2){
							if(B[i1-1] != B[i2-1]){
								TN++;
							}else{
								FN++;
							}
						}
					}

				}
			}
		}


		double R = TP * 1.0/(TP + FN);
		return R;
	}
	/*
     * @param A,B
     * @return RankIndex
     */
	public static double RI(int[] A,int[] B){

		double value = 0.0;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		int[] commonNo = commNum(clusterA,clusterB);//得到A中每个簇中聚类正确的数目。
		int P = 0;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			P = P + combination(entryA.getValue().size(),2);
			TP = TP + combination(commonNo[i],2);
			i++;
		}

		FP = P - TP;

		itA = clusterA.entrySet().iterator();
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();

			Iterator<Map.Entry<Integer,Set<Integer>>> itA2 = clusterA.entrySet().iterator();
			while(itA2.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryA2 = itA2.next();
				if(entryA != entryA2){
					Set<Integer> s1 = entryA.getValue();
					Set<Integer> s2 = entryA2.getValue();
					for(Integer i1 :s1){
						for(Integer i2:s2){
							if(B[i1-1] != B[i2-1]){
								TN++;
							}else{
								FN++;
							}
						}
					}

				}
			}
		}
		value = (TP + TN)*1.0/(TP + FP + FN + TN);

		return value;
	}

	/*
     * F值，是对精度和召回率的平衡，
     * @param A:评估对象。B：评估标准；beta：均衡参数
     * @return F值
     */
	public static double F_score(int[] A,int[] B){

		double beta = 1.0;
		double value = 0.0;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		int[] commonNo = commNum(clusterA,clusterB);//得到A中每个簇中聚类正确的数目。
		int allP = 0;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			allP = allP + combination(entryA.getValue().size(),2);
			TP = TP + combination(commonNo[i],2);
			i++;
		}

		FP = allP - TP;

		itA = clusterA.entrySet().iterator();
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();

			Iterator<Map.Entry<Integer,Set<Integer>>> itA2 = clusterA.entrySet().iterator();
			while(itA2.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryA2 = itA2.next();
				if(entryA != entryA2){
					Set<Integer> s1 = entryA.getValue();
					Set<Integer> s2 = entryA2.getValue();
					for(Integer i1 :s1){
						for(Integer i2:s2){
							if(B[i1-1] != B[i2-1]){
								TN++;
							}else{
								FN++;
							}
						}
					}

				}
			}
		}

		double P = TP*1.0/(TP + FP);
		double R = TP * 1.0/(TP + FN);
		value = (beta*beta + 1)*P * R/(beta*beta*P + R);
		return value;
	}

	public static double Jaccard(int[] A,int[] B){

		double value = 0.0;
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		int[] commonNo = commNum(clusterA,clusterB);//得到A中每个簇中聚类正确的数目。
		int allP = 0;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();
		int i = 0;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			allP = allP + combination(entryA.getValue().size(),2);
			TP = TP + combination(commonNo[i],2);
			i++;
		}

		FP = allP - TP;

		itA = clusterA.entrySet().iterator();
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();

			Iterator<Map.Entry<Integer,Set<Integer>>> itA2 = clusterA.entrySet().iterator();
			while(itA2.hasNext()){
				Map.Entry<Integer,Set<Integer>> entryA2 = itA2.next();
				if(entryA != entryA2){
					Set<Integer> s1 = entryA.getValue();
					Set<Integer> s2 = entryA2.getValue();
					for(Integer i1 :s1){
						for(Integer i2:s2){
							if(B[i1-1] != B[i2-1]){
								TN++;
							}else{
								FN++;
							}
						}
					}

				}
			}
		}


		value = TP * 1.0 / (TP + FP + FN);
		return value;
	}
	public static double NMI(int[] A,int[] B){
		Map<Integer,Set<Integer>> clusterA = clusterDistri(A);//得到聚类结果A的类分布
		Map<Integer,Set<Integer>> clusterB = clusterDistri(B);//得到聚类B（标准）的类分布
		Iterator<Map.Entry<Integer,Set<Integer>>> itA = clusterA.entrySet().iterator();

		Iterator<Map.Entry<Integer,Set<Integer>>> itB = clusterB.entrySet().iterator();

		Set<Set<Integer>> partitionF = new HashSet<Set<Integer>>();
		Set<Set<Integer>> partitionR = new HashSet<Set<Integer>>();
		int nodeCount = B.length;
		while(itA.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryA = itA.next();
			Set<Integer> setA = entryA.getValue();
			partitionF.add(setA);
			setA = null;
			entryA = null;
		}


		while(itB.hasNext()){
			Map.Entry<Integer,Set<Integer>> entryB = itB.next();
			Set<Integer> setB = entryB.getValue();
			partitionR.add(setB);
			setB = null;
			entryB = null;
		}
		return computeNMI(partitionF,partitionR,nodeCount);
	}
	public static double computeNMI(Set<Set<Integer>> partitionF,
									Set<Set<Integer>> partitionR,int nodeCount) {
		int[][] XY = new int[partitionR.size()][partitionF.size()];
		int[] X = new int[partitionR.size()];
		int[] Y = new int[partitionF.size()];
		int i = 0;
		int j = 0;

		for (Set<Integer> com1 : partitionR) {
			j = 0;

			for (Set<Integer> com2 : partitionF) {

				XY[i][j] = intersect(com1, com2);//待测结果第i个簇和标准结果第j个簇的共有元素个数
				X[i] += XY[i][j];//待测结果第i个簇与所有标准结果簇的公共元素个数（感觉就是第i个簇的元素个数）
				Y[j] += XY[i][j];//标准结果簇第j个簇的元素个数（）

				j++;
			}
			i++;
		}
		int N = nodeCount;
		double Ixy = 0;
		double Ixy2 = 0;
		for (i = 0; i < partitionR.size(); i++) {
			for (j = 0; j < partitionF.size(); j++) {
				if (XY[i][j] > 0) {
					Ixy += ((double) XY[i][j] / N)
							* (Math.log((double) XY[i][j] * N / (X[i] * Y[j])) / Math
							.log(2.0));
//                  Ixy2 = (float) (Ixy2 + -2.0D * XY[i][j]
//                          * Math.log(XY[i][j] * N / X[i] * Y[j]));
				}
			}
		}
//      System.out.println(Ixy2);
//      double denom = 0.0F;
//      for (int ii = 0; ii < X.length; ++ii)
//          denom = (double) (denom + X[ii] * Math.log(X[ii] / N));
//      for (int jj = 0; jj < Y.length; ++jj) {
//          denom = (double) (denom + Y[jj] * Math.log(Y[jj] / N));
//      }
//
//      System.out.println(denom);
//      double M = (Ixy / denom);
//
//      return M;

		double Hx = 0;
		double Hy = 0;
		for (i = 0; i < partitionR.size(); i++) {
			if (X[i] > 0)
				Hx += h((double) X[i] / N);
		}
		for (j = 0; j < partitionF.size(); j++) {
			if (Y[j] > 0)
				Hy += h((double) Y[j] / N);
		}

		double InormXY = Ixy / Math.sqrt(Hx * Hy);
		return InormXY;
	}
	private static double h(double p) {
		return -p * (Math.log(p) / Math.log(2.0));
	}
	/*
     * 两个集合的公共元素个数
     */
	private static int intersect(Set<Integer> com1, Set<Integer> com2) {
		int num = 0;
		for (Integer v1 : com1) {
			if (com2.contains(v1))
				num++;
		}
		return num;
	}
	/*
     * C(m,n)=m取n
     */
	public static int combination(int m,int n){
		int result = 1;
		if(m < n){
			return -1;
		}
		result = factorial(m)/(factorial(n)*factorial(m-n));

		return result;
	}

	public static int factorial(int m){

		if((m == 1) || (m == 0)){
			return 1;
		}else if(m < 0){
			return -1;
		}else{
			return m*factorial(m-1);
		}
	}
}
