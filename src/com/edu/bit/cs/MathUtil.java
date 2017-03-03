package com.edu.bit.cs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian;

public class MathUtil
{
	//һЩ�������õ��ĳ���ֵ
	private static final double ZERO = 0.000001;

	//ͨ��EM�㷨�õ�GMM
	public static ICGTGaussianMixtureModel getGMMByEM(JavaRDD<Vector> samples, int k)
	{
		return new ICGTGaussianMixtureModel(samples, k);
	}

	//�ϲ�������˹���ģ��Ϊһ����˹���ģ��
	public static ICGTGaussianMixtureModel mergeGMM(ICGTGaussianMixtureModel icgtGMMA,ICGTGaussianMixtureModel icgtGMMB) throws Exception
	{
		if(icgtGMMA == null || icgtGMMB == null)
		{
			throw new Exception("ICGTGaussianMixtureModel is null");
		}
		int numOfGassiansA = icgtGMMA.numOfGaussians();
		int numOfGassiansB = icgtGMMB.numOfGaussians();
		int numOfGaussians = numOfGassiansA + numOfGassiansB;
		long numOfSamplesA = icgtGMMA.numOfSamples();
		long numOfSamplesB = icgtGMMB.numOfSamples();
		long numOfSamples = numOfSamplesA + numOfSamplesB;
		double weights[] = new double[numOfGaussians];
		MultivariateGaussian[] gaussians = new MultivariateGaussian[numOfGaussians];
		for(int i = 0; i < numOfGaussians; i++)
		{
			long numOfSamplesTmp;
			if(i < numOfGassiansA)
			{
				numOfSamplesTmp = (long)((double)numOfSamplesA * icgtGMMA.weight(i) + 0.5);
				gaussians[i] = new MultivariateGaussian(icgtGMMA.gaussian(i).mu(),icgtGMMA.gaussian(i).sigma());
				weights[i] = (double)numOfSamplesTmp / (double)numOfSamples;
			}
			else
			{
				numOfSamplesTmp = (long)((double)numOfSamplesB * icgtGMMB.weight(i-numOfGassiansA) + 0.5);
				gaussians[i] = new MultivariateGaussian(icgtGMMB.gaussian(i-numOfGassiansA).mu(),icgtGMMB.gaussian(i-numOfGassiansA).sigma());
				weights[i] = (double)numOfSamplesTmp / (double)numOfSamples;
			}
		}

		return new ICGTGaussianMixtureModel(weights,gaussians,numOfSamples);
	}
/*
	public static  double[] calculateMu(long sum,double[] arrMu,ICGTNode node)
	{
		//�ж��Ƿ�Ϊ��һ�μ����ֵ
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
			//����ǵ�һ�μ��㣬ֱ�Ӱѽ��ľ�ֵȫ����ֵ��ȥ����
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
		//�ж��Ƿ�Ϊ��һ�μ����ֵ
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
			//����ǵ�һ�μ��㣬ֱ�Ӱѽ��ľ�ֵȫ����ֵ��ȥ����
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
			//���¸��ڵ��ֵ�ͷ�����ǹ�ʽ��û�Ƶ�����
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

	//KLɢ�Ⱦ�����㹫ʽ
	public static double KLDivergenceDistance (MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
	{
		double result = 0;
		int dimension = gaussianA.mu().size();

		for (int i = 0; i < dimension; i++)
		{
			//ԭ������ΪЭ���java��ΪЭ�������
			if (gaussianB.sigma().apply(i, i) == 0)
			{
				gaussianB.sigma().update(i, i, ZERO);
			}
			if (gaussianA.sigma().apply(i, i) == 0)
			{
				result += gaussianA.sigma().apply(i, i) / gaussianB.sigma().apply(i, i) + Math.pow((gaussianA.mu().apply(i) - gaussianB.mu().apply(i)), 2.0) / gaussianB.sigma().apply(i, i);
			}
			else
			{
				result += gaussianA.sigma().apply(i, i) / gaussianB.sigma().apply(i, i) + Math.pow((gaussianA.mu().apply(i) - gaussianB.mu().apply(i)), 2.0) / gaussianB.sigma().apply(i, i)
						- Math.log(Math.abs(gaussianA.sigma().apply(i, i) / gaussianB.sigma().apply(i, i)));
			}
		}
		result -= dimension;
		result *= 0.5;
		double temp = 1 / (1 + result);
		return temp;
	}


	//����������˹���ģ�͵ľ���
	public static  double GQFDistance(ICGTGaussianMixtureModel icgtGMMA,ICGTGaussianMixtureModel icgtGMMB)
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
		double[][] A = new double[800][800];

		for (int i = 0; i < numOfGaussians; i++)
		{
			for (int j = 0; j < numOfGaussians; j++)
			{
				A[i][j] = MathUtil.KLDivergenceDistance(gaussians[i], gaussians[j]);
			}
		}

		double result1[] = new double[800];
		double result = 0;
		for(int i = 0 ; i < 800 ;++i){
			result1[i] = 0;
		}
		for (int i = 0; i < numOfGaussians; ++i)
		{
			for (int j = 0; j < numOfGaussians; ++j)
			{
				result1[i] += weights[j] * A[j][i];
			}
		}

		for (int i = 0; i < numOfGaussians; ++i)
		{
			result += result1[i] * weights[i];
		}
		if (result < 0)
		{
			result = 0;
		}
		return Math.sqrt(result);
	}

	//����������˹ģ�͵�ŷʽ�������
	public static  double eulcideanDistance(MultivariateGaussian gaussianA,MultivariateGaussian gaussianB)
	{
		double distance = 0;
		int dimension = gaussianA.mu().size();
		for (int i = 0; i < dimension ; ++i)
		{
			double temp = gaussianA.mu().apply(i) - gaussianB.mu().apply(i);
			distance += temp * temp;
		}
		distance = Math.sqrt(distance); //ŷ�Ͼ���
		return distance;
	}

}
