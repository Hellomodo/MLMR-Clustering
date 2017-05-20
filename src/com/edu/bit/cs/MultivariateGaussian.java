package com.edu.bit.cs;

import scala.Array;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by 林 on 2017/3/3.
 */
public class MultivariateGaussian implements Serializable
{
    private double[] _mean;
    private double[] _cov;

    private long _numOfSamples;

    private static double _covInit = 0.1;

    public MultivariateGaussian(double[] mean, double[] cov, long numOfSamples)
    {
        _mean = mean;
        _cov = cov;
        _numOfSamples = numOfSamples;
    }


    public MultivariateGaussian(Sample sample)
    {
        _mean = sample.variables();
        _cov = new double[_mean.length];
        Arrays.fill(_cov,_covInit);
        _numOfSamples = 1;
    }


    public double[] mean()
    {
        return _mean;
    }

    public double[] cov()
    {
        return _cov;
    }

    public int dimension()
    {
        return _mean.length;
    }

    public long numOfSamples()
    {
        return _numOfSamples;
    }
}
