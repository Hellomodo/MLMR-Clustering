package com.edu.bit.cs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public class MathUtil
{
	//一些运算里用到的常量值
	private static final double ZERO = 0.000001;

/*
	public static ICGTGaussianMixtureModel getGMMByEM(JavaRDD<Vector> samples, int k)
	{
		return new ICGTGaussianMixtureModel(samples, k);
	}*/


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

	/*
	//合并两个高斯混合模型为一个高斯混合模型
	public static GaussianMixtureModel mergeGMM(GaussianMixtureModel gmmA,GaussianMixtureModel gmmB) throws Exception
	{
		if(gmmA == null || gmmB == null)
		{
			throw new Exception("ICGTGaussianMixtureModel is null");
		}
		int numOfGassiansA = gmmA.numOfGaussians();
		int numOfGassiansB = gmmB.numOfGaussians();
		int numOfGaussians = numOfGassiansA + numOfGassiansB;
		long numOfSamplesA = gmmA.numOfSamples();
		long numOfSamplesB = gmmB.numOfSamples();
		long numOfSamples = numOfSamplesA + numOfSamplesB;
		double weights[] = new double[numOfGaussians];
		MultivariateGaussian[] gaussians = new MultivariateGaussian[numOfGaussians];
		for(int i = 0; i < numOfGaussians; i++)
		{
			long numOfSamplesTmp;
			if(i < numOfGassiansA)
			{
				gaussians[i] = new MultivariateGaussian(gmmA.gaussian(i).mu(),gmmA.gaussian(i).sigma());
				weights[i] = (double)numOfSamplesTmp / (double)numOfSamples;
			}
			else
			{
				numOfSamplesTmp = (long)((double)numOfSamplesB * gmmB.weight(i-numOfGassiansA) + 0.5);
				gaussians[i] = new MultivariateGaussian(gmmB.gaussian(i-numOfGassiansA).mu(),gmmB.gaussian(i-numOfGassiansA).sigma());
				weights[i] = (double)numOfSamplesTmp / (double)numOfSamples;
			}
		}

		return new GaussianMixtureModel(weights,gaussians,numOfSamples);
	}
	*/
/*
	public static  double[] calculateMu(long sum,double[] arrMu,ICGTNode node)
	{
		//判断是否为第一次计算均值
		if(sum < 0)
		{
			try
			{
				throw new Exception("Sum can't be negtive!");
			} catch (Exception e)
			{
				e.printStackTrace();
			}
			return null;
		}
		double[] res = new double[node.dimension];
		if(sum == 0)
		{
			//如果是第一次计算，直接把结点的均值全部赋值过去就行
			for(int i = 0; i < arrMu.length ; ++i)
			{
				res[i] = node.mu.apply(i);
			}
		}
		else{
			for(int i = 0; i < arrMu.length ; ++i)
			{
				res[i] = (double)(arrMu[i] * sum + node.mu.apply(i))/(double)(sum+node.dataNum);
			}
		}
		return res;
	}
*/
/*
	public static  double[] calculateSigma(long sum,double[] arrSigma,ICGTNode node){
		//判断是否为第一次计算均值
		if(sum < 0){
			try {
				throw new Exception("Sum can't be negtive!");
			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}
		double[] res = new double[node.dimension*node.dimension];
		if(sum == 0)
		{
			//如果是第一次计算，直接把结点的均值全部赋值过去就行
			for(int i = 0; i < node.dimension ; ++i)
			{
				for(int j = 0; j < node.dimension ; ++j)
				{
					res[i*node.dimension+j] = node.sigma.apply(i,j);
				}
			}
		}
		else
		{
			//更新父节点均值和方差，但是公式还没推导出来
		}
		return res;
	}
*/
	public static void warshall(boolean[][] mat)
	{
		int num = mat.length;
		for (int i = 0; i < num; ++i)
		{
			for (int j = 0; j < num; ++j)
			{
				if (mat[i][j])
				{
					for (int k = 0; k < num; ++k)
					{
						if (mat[k][i])
						{
							mat[k][j] = true;
						}
					}
				}
			}
		}
	}

	//KL散度距离计算公式
	public static double KLDivergenceDistance (MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
	{
		double result = 0;
		int dimension = gaussianA.dimension();
		double[] meanA = gaussianA.mean();
		double[] covA = gaussianA.cov();
		double[] meanB = gaussianB.mean();
		double[] covB = gaussianB.cov();
		for (int i = 0; i < dimension; i++)
		{
			//原来代码为协方差，java中为协方差矩阵
			if (covB[i] == 0)
			{
				covB[i] = ZERO;
			}
			if (covA[i] == 0)
			{
				result += covA[i] / covB[i]+ Math.pow((meanA[i] - meanB[i]), 2.0) / covB[i];
			}
			else
			{
				result += covA[i] / covB[i] + Math.pow((meanA[i] - meanB[i]), 2.0) / covB[i]
						- Math.log(Math.abs(covA[i] / covB[i]));
			}
		}
		result -= dimension;
		result *= 0.5;
		double temp = 1 / (1 + result);
		return temp;
	}


	//计算两个高斯混合模型的距离
	public static  double GQFDistance(GaussianMixtureModel icgtGMMA,GaussianMixtureModel icgtGMMB)
	{
		int numOfGassiansA = icgtGMMA.numOfGaussians();
		int numOfGassiansB = icgtGMMB.numOfGaussians();
		int numOfGaussians = numOfGassiansA + numOfGassiansB;

		double[] weights = new double[numOfGaussians];
		System.arraycopy(icgtGMMA.weights(), 0, weights, 0, numOfGassiansA);
		System.arraycopy(icgtGMMB.weights(), 0,weights, numOfGassiansA, numOfGassiansB);

		MultivariateGaussian[] gaussians = new MultivariateGaussian[numOfGaussians];
		System.arraycopy(icgtGMMA.gaussians(), 0, gaussians, 0, numOfGassiansA);
		System.arraycopy(icgtGMMB.gaussians(), 0,gaussians, numOfGassiansA, numOfGassiansB);

		double[][] A = new double[numOfGaussians][numOfGaussians];
		for (int i = 0; i < numOfGaussians; i++)
		{
			for (int j = 0; j < numOfGaussians; j++)
			{
				A[i][j] = MathUtil.KLDivergenceDistance(gaussians[i], gaussians[j]);
			}
		}

		double result = 0;

		for (int i = 0; i < numOfGaussians; ++i)
		{
			for (int j = 0; j < numOfGaussians; ++j)
			{
				result += weights[j] * A[j][i] * weights[i];
			}
		}
		result = result < 0 ? 0 : result;

		return Math.sqrt(result);
	}

	//计算两个高斯模型的欧式距离距离
	public static  double eulcideanDistance(MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
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

}
