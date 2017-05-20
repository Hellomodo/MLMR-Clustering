package com.edu.bit.cs;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.*;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Serializable;

import javax.swing.*;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;


public class Main implements Serializable{

    static final String _urlMaster= "spark://10.108.6.128:7077";
    static final String _urlHDFS = "D:\\LsyTestProj\\ICGTClustering\\";
    static final String _urlLocalHome= "D:\\LsyTestProj\\ICGTClustering\\";
    static JavaSparkContext _javaSparkContext = null;
    static final String path = _urlHDFS + "syn_2_2_10000k.txt";

    public static void main(String[] args) throws Exception
    {

        System.setProperty("hadoop.home.dir", "D:\\ProjSoftware\\hadoop-2.7.3");
        SparkConf  _sparkConf = new SparkConf().setAppName("GaussianMixture Example").setMaster("local[1]")
                .setJars(new String[]{"D:\\LsyTestProj\\ICGTClustering\\out\\artifacts\\ICGTClustering_jar\\ICGTClustering.jar"});
        _javaSparkContext = new JavaSparkContext(_sparkConf);

        Main app = new Main();
        app.runPICGT();
    }

    public void runPICGT()
    {
        JavaRDD<String> data = _javaSparkContext.textFile(path);
        JavaRDD<Sample> rddSample = data.map(
                new Function<String, Sample>()
                {
                    public Sample call(String s)
                    {
                        // String[] sarray = s.trim().split("\\s+");
                        String[] sarray = s.trim().split("\t");
                        double[] values = new double[sarray.length - 1];
                        for (int i = 0; i < sarray.length - 1; i++)
                            values[i] = Double.parseDouble(sarray[i]);
                        Sample sample = new Sample( values, Integer.parseInt(sarray[sarray.length - 1]) );
                        return sample;
                    }
                }
        );

        long start = System.currentTimeMillis();	// 记录起始时间

        //rddSample.repartition(3);
        List<ICGTNode> listSubtrees = rddSample.mapPartitions(
                new FlatMapFunction<Iterator<Sample>,ICGTNode>()
                {
                    public Iterable<ICGTNode> call(Iterator<Sample> data)
                    {
                        ICGTClustering icgtClustering = new ICGTClustering();
                        icgtClustering.run(data);
                        return icgtClustering.getFirstLayer();
                    }
                }
        ).collect();

        //ICGTClustering icgtClustering = new ICGTClustering();
        // icgtClustering.run(parsedData.collect().iterator());

        //icgtClustering.showResults();
        //ICGTClustering finalClustering = new ICGTClustering(icgtClustering.getFirstLayer().iterator());

        ICGTClustering finalClustering = new ICGTClustering(listSubtrees.iterator());

        long end = System.currentTimeMillis();		// 记录结束时

        System.out.println(end-start);				// 相减得出运行时间
        finalClustering.showResults();
        List<Sample> samples = finalClustering.getResults();
        System.out.println("Purity:" + MathUtil.ClusEvaluate("Purity",samples));


    }

    public void runKmeans()
    {
        JavaRDD<String> data = _javaSparkContext.textFile(path);

        JavaRDD<Sample> rddSample = data.map(
                new Function<String, Sample>()
                {
                    public Sample call(String s)
                    {
                       // String[] sarray = s.trim().split("\\s+");
                        String[] sarray = s.trim().split("\t");
                        double[] values = new double[sarray.length - 1];
                        for (int i = 0; i < sarray.length - 1; i++)
                            values[i] = Double.parseDouble(sarray[i]);
                        Sample sample = new Sample( values, Integer.parseInt(sarray[sarray.length - 1]) );
                        return sample;
                    }
                }
        );

        JavaRDD<Vector> rddVector = rddSample.map(
                new Function<Sample, Vector>()
                {
                    public Vector call(Sample s)
                    {
                        return Vectors.dense(s.variables());
                    }
                }
        );
        int numClusters = 2;
        int numIterations = 20;
        int numTimes = 10;

        long start = System.currentTimeMillis();	// 记录起始时间

        final KMeansModel clusters = KMeans.train(rddVector.rdd(),numClusters,numIterations,numTimes);
        List<Sample> samples = rddSample.map(
                new Function<Sample, Sample>()
                {
                    public Sample call(Sample s)
                    {
                        Vector v = Vectors.dense(s.variables());
                        int predict = clusters.predict(v);
                        s.setPridict(predict + 1);
                        return s;
                    }
                }
        ).collect();
        long end = System.currentTimeMillis();		// 记录结束时间

        System.out.println(end-start);				// 相减得出运行时间
        //System.out.println("Purity:" + MathUtil.ClusEvaluate("Purity",samples));

        final JFrame frame = new JFrame("Point Data Rendering");
        ICGTPanel panel = new ICGTPanel();
        panel.displayClusters(samples);
        frame.setContentPane(panel);
        frame.pack();
        frame.setVisible(true);
        frame.repaint();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }


    public void runGMM()
    {
        JavaRDD<String> data = _javaSparkContext.textFile(path);

        JavaRDD<Sample> rddSample = data.map(
                new Function<String, Sample>()
                {
                    public Sample call(String s)
                    {
                        // String[] sarray = s.trim().split("\\s+");
                        String[] sarray = s.trim().split("\t");
                        double[] values = new double[sarray.length - 1];
                        for (int i = 0; i < sarray.length - 1; i++)
                            values[i] = Double.parseDouble(sarray[i]);
                        Sample sample = new Sample( values, Integer.parseInt(sarray[sarray.length - 1]) );
                        return sample;
                    }
                }
        );

        JavaRDD<Vector> rddVector = rddSample.map(
                new Function<Sample, Vector>()
                {
                    public Vector call(Sample s)
                    {
                        return Vectors.dense(s.variables());
                    }
                }
        );
        int numClusters = 2;
        long start = System.currentTimeMillis();	// 记录起始时间
        final org.apache.spark.mllib.clustering.GaussianMixtureModel clusters = new GaussianMixture().setK(numClusters).run(rddVector.rdd());

        List<Sample> samples = rddSample.map(
                new Function<Sample, Sample>()
                {
                    public Sample call(Sample s)
                    {
                        Vector v = Vectors.dense(s.variables());
                        int predict = clusters.predict(v);
                        s.setPridict(predict + 1);
                        return s;
                    }
                }
        ).collect();
        long end = System.currentTimeMillis();		// 记录结束时间
        System.out.println(end-start);				// 相减得出运行时间
        System.out.println("Purity:" + MathUtil.ClusEvaluate("Purity",samples));

        final JFrame frame = new JFrame("Point Data Rendering");
        ICGTPanel panel = new ICGTPanel();
        panel.displayClusters(samples);
        frame.setContentPane(panel);
        frame.pack();
        frame.setVisible(true);
        frame.repaint();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }
}