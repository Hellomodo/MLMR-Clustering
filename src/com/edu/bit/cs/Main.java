package com.edu.bit.cs;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.Iterator;
import java.util.LinkedList;


public class Main{

    public static void main(String[] args) throws Exception
    {
        String _urlMaster= "spark://10.108.6.128:7077";
        String _urlHDFS = "hdfs://10.108.6.128:9000/";
        System.setProperty("hadoop.home.dir", "D:\\ProjSoftware\\hadoop-2.7.3");
        SparkConf  _sparkConf = new SparkConf().setAppName("GaussianMixture Example").setMaster("local")
                .setJars(new String[]{"D:\\LsyTestProj\\ICGTClustering\\out\\artifacts\\ICGTClustering_jar\\ICGTClustering.jar"});
        JavaSparkContext _javaSparkContext = new JavaSparkContext(_sparkConf);

        // Load and parse data
        String path = _urlHDFS + "ICGT_Samples/Jain1.txt";
        JavaRDD<String> data = _javaSparkContext.textFile(path);
        JavaRDD<Vector> parsedData = data.map(
                new Function<String, Vector>()
                {
                    public Vector call(String s)
                    {
                        String[] sarray = s.trim().split("\t");
                        double[] values = new double[sarray.length-1];
                        for (int i = 0; i < sarray.length-1; i++)
                            values[i] = Double.parseDouble(sarray[i]);
                        return Vectors.dense(values);
                    }
                }
        );

        parsedData.cache();
        ICGTClustering clustering = new ICGTClustering();
        long start = System.currentTimeMillis();	// ��¼��ʼʱ��
        clustering.run(parsedData);
        long end = System.currentTimeMillis();		// ��¼����ʱ��
        System.out.println(end-start);				// ����ó�����ʱ��
        start = System.currentTimeMillis();	// ��¼��ʼʱ��
        clustering.showResults();
        end = System.currentTimeMillis();		// ��¼����ʱ��
        System.out.println(end-start);				// ����ó�����ʱ��
        //ͨ���޸�showResults�����޸ľ�����Ժ���ʾ��ʽ
        //clustering.showResults(parsedData);
    }
}