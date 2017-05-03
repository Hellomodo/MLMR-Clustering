package com.edu.bit.cs;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;


public class Main{

    public static void main(String[] args) throws Exception
    {
        String _urlMaster= "spark://10.108.6.128:7077";
        String _urlHDFS = "D:\\LsyTestProj\\ICGTClustering\\";
        System.setProperty("hadoop.home.dir", "D:\\ProjSoftware\\hadoop-2.7.3");
        SparkConf  _sparkConf = new SparkConf().setAppName("GaussianMixture Example").setMaster("local")
                .setJars(new String[]{"D:\\LsyTestProj\\ICGTClustering\\out\\artifacts\\ICGTClustering_jar\\ICGTClustering.jar"});
        JavaSparkContext _javaSparkContext = new JavaSparkContext(_sparkConf);

        // Load and parse data
        String path = _urlHDFS + "BIRCH1.txt";
        JavaRDD<String> data = _javaSparkContext.textFile(path);
        JavaRDD<Vector> parsedData = data.map(
                new Function<String, Vector>()
                {
                    public Vector call(String s)
                    {
                        String[] sarray = s.trim().split("\\s+");
                        //String[] sarray = s.trim().split("\t");
                        double[] values = new double[sarray.length - 1 ];
                        for (int i = 0; i < sarray.length - 1 ; i++)
                            values[i] = Double.parseDouble(sarray[i]);
                        return Vectors.dense(values);
                    }
                }
        );

        long start = System.currentTimeMillis();	// 记录起始时间

        //parsedData = parsedData.repartition(2);
/*
        List<ICGTNode> listSubtrees = parsedData.mapPartitions(
                new FlatMapFunction<Iterator<Vector>,ICGTNode>()
                {
                    public Iterable<ICGTNode> call(Iterator<Vector> data)
                    {
                        ICGTClustering icgtClustering = new ICGTClustering();
                        icgtClustering.run(data);
                        return icgtClustering.getFirstLayer();
                    }
                }
        ).collect();
*/
        ICGTClustering icgtClustering = new ICGTClustering();
        icgtClustering.run(parsedData.collect().iterator());

       ICGTClustering finalClustering = new ICGTClustering(icgtClustering.getFirstLayer().iterator());

        finalClustering.showResults();

        long end = System.currentTimeMillis();		// 记录结束时
        System.out.println(end-start);				// 相减得出运行时间
    }
}