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

    //一些运算里用到的常量值
    private static final double ZERO = 0.000001;
    
    private static final double G_CONNECT_THRESHOLD = 3.7;
    private static final int DISTANCE_THRESHOLD = 0;
    private static final double GMM_CONNECT_THRESHOLD = 2;

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
        Arrays.fill(_cov,ZERO);
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
