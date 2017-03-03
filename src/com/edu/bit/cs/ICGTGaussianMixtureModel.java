package com.edu.bit.cs;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian;
//高斯混合模型适配器,增加数据个数属性




import breeze.linalg.Counter;

public class ICGTGaussianMixtureModel
{
	private GaussianMixtureModel _gmm;
	private long _numOfSamples;

	public ICGTGaussianMixtureModel(double[] weights, MultivariateGaussian[] gaussian, long numOfSamples)
	{
		_gmm = new GaussianMixtureModel(weights, gaussian);
		_numOfSamples = numOfSamples;
	}

	public ICGTGaussianMixtureModel(MultivariateGaussian icgtGaussian, long numOfSamples)
	{
		double[] weights = new double[1];
		weights[0] = 1;
		MultivariateGaussian[] icgtGaussians = new MultivariateGaussian[1];
		icgtGaussians[0] = icgtGaussian;
		_gmm = new GaussianMixtureModel(weights,icgtGaussians);
		_numOfSamples = numOfSamples;
	}

	public ICGTGaussianMixtureModel(JavaRDD<Vector> samples, int numOfCluster)
	{
		_gmm = new GaussianMixture().setK(numOfCluster).run(samples.rdd());
		_numOfSamples = samples.count();
	}

	public long numOfSamples(int index)
	{
		return (long)( (double)_numOfSamples * this.weight(index) + 0.5 );
	}

	public long numOfSamples()
	{
		int numOfSamles = 0;
		for(int i = 0; i < this.numOfGaussians(); i++)
		{
			numOfSamles += this.numOfSamples(i);
		}
		return numOfSamles;
	}

	public int numOfGaussians()
	{
		return _gmm.k();
	}

	public int dimension()
	{
		return _gmm.gaussians()[0].mu().size();
	}

	public MultivariateGaussian[] gaussians()
	{
		return _gmm.gaussians();
	}

	public MultivariateGaussian gaussian(int index)
	{
		return _gmm.gaussians()[index];
	}

	public double[] weights()
	{
		return _gmm.weights();
	}

	public double weight(int index)
	{
		return _gmm.weights()[index];
	}

	public List<Integer> predict(JavaRDD<Vector> samples)
	{
		return _gmm.predict(samples).collect();
	}
}
