package com.edu.bit.cs;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public class GaussianMixtureModel
{

	private MultivariateGaussian[] _gaussians;
	private double[] _weights;
	private long _numOfSamples;

	public GaussianMixtureModel(MultivariateGaussian gaussian)
	{
		_gaussians = new MultivariateGaussian[1];
		_gaussians[0] = gaussian;
		_numOfSamples = 1;
		_weights = new double[1];
		_weights[0] = 1;
	}

	public GaussianMixtureModel(MultivariateGaussian[] gaussian)
	{
		_gaussians = gaussian;
		_weights = new double[ _gaussians.length];
		_numOfSamples = 0;
		for(int i = 0; i < _gaussians.length; i++)
		{
			_numOfSamples += _gaussians[i].numOfSamples();
		}
		for(int i = 0; i < _gaussians.length; i++)
		{
			_weights[i] = _gaussians[i].numOfSamples()/(double)_numOfSamples;
		}

	}
	public long numOfSamples(int index)
	{
		return _gaussians[index].numOfSamples();
	}

	public long numOfSamples()
	{
		return _numOfSamples;
	}

	public int numOfGaussians()
	{
		return _gaussians.length;
	}

	public MultivariateGaussian[] gaussians()
	{
		return _gaussians;
	}

	public MultivariateGaussian gaussian(int index)
	{
		return _gaussians[index];
	}


	public double[] weights()
	{
		return _weights;
	}

	public double weight(int index)
	{
		return _weights[index];
	}
}
