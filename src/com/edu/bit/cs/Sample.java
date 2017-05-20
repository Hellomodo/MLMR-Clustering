package com.edu.bit.cs;
;import java.io.Serializable;

/**
 * Created by 林 on 2017/3/3.
 */
public class Sample implements Serializable
{
    private double[] _variables;

    private int _label;

    private int _pridict;

    public Sample(double[] variables)
    {
        _variables = variables;
    }

    public Sample(double[] variables,int label)
    {
        _variables = variables;
        _label = label;
    }

    public void setLabel(int label)
    {
        _label = label;
    }

    public int getLabel()
    {
        return _label;
    }

    public void setPridict(int pridict)
    {
        _pridict = pridict;
    }

    public int getPridict()
    {
        return _pridict;
    }

    public double variable(int index)
    {
        return _variables[index];
    }

    public double[] variables()
    {
        return _variables;
    }


}
