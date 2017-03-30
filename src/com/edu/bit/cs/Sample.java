package com.edu.bit.cs;

import com.sun.xml.bind.v2.model.core.ID;
import com.sun.xml.internal.ws.developer.Serialization;
import org.apache.spark.mllib.linalg.Vector;

import java.io.Serializable;

/**
 * Created by 林 on 2017/3/3.
 */
public class Sample implements Serializable
{
    private double[] _variables;

    private int _label;

    public Sample(Vector variables)
    {
        _variables = variables.toArray();
    }

    public void setLabel(int label)
    {
        _label = label;
    }

    public int getLabel()
    {
        return _label;
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
