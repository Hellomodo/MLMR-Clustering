package com.edu.bit.cs;

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
		//return resultAtoB;
		return Math.atan(resultAtoB) * 2 / Math.PI;
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

		double result = 0;
		for (int i = 0; i < numOfGaussians; i++)
		{
			for (int j = 0; j < numOfGaussians; j++)
			{
				double Aij = Math.min( MathUtil.KLDivergence(gaussians[i], gaussians[j]), MathUtil.KLDivergence(gaussians[j], gaussians[i]));
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

}
