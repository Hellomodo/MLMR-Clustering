package com.edu.bit.cs;
import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

import org.apache.spark.mllib.linalg.Vector;


public class ICGTPanel extends JPanel
{
	List<Vector> _data;
	List<Integer> _lable;
	Map<Integer,Color> _tabColor;

	private int margin = 20;
	private int _lengthX = 1200;
	private int _lengthY = 800;
	private double _minX, _maxX;
	private double _minY, _maxY;
	private double _spanX, _spanY;
	private int _lengthP = 5;

	public Point coordTransform(double x, double y)
	{
		Point p = new Point();
		p.x = (int)( (x - _minX) * _lengthX  / _spanX  ) + margin;
		p.y = (int)( (y - _minY) * _lengthY / _spanY ) + margin;
		return p;
	}
	@Override
	public void paintComponent(final Graphics g)
	{
		super.paintComponent(g);

		Random rand =new Random(25);

		Iterator itData = _data.iterator();
		Iterator itLable = _lable.iterator();
		while(itData.hasNext())
		{
			Vector data = (Vector)itData.next();
			Integer lable = (Integer)itLable.next();
			if(! _tabColor.containsKey(lable))
			{
				_tabColor.put(lable,new Color((int)(Math.random()*256),(int)(Math.random()*256),(int)(Math.random()*256)));

			}
			g.setColor(_tabColor.get(lable));
			Point point = coordTransform(data.apply(0), data.apply(1));
			g.fillOval((int)point.getX(),(int)point.getY(), _lengthP, _lengthP);
		}
	}

	public void displayClusters(List<Vector> data, List<Integer> lable)
	{
		_data = data;
		_lable = lable;
		_tabColor = new HashMap<Integer, Color>();

		_minX = Double.MAX_VALUE;
		_maxX = 0;
		_minY =  Double.MAX_VALUE ;
		_maxY = 0;

		for(Vector p : _data)
		{
			double x = p.apply(0), y = p.apply(1);
			_minX = _minX < x ? _minX : x;
			_maxX = _maxX > x ? _maxX : x;
			_minY = _minY < y ? _minY : y;
			_maxY = _maxY > y ? _maxY : y;
		}
		_spanX = _maxX - _minX;
		_spanY = _maxY - _minY;
		this.setPreferredSize( new Dimension(_lengthX + margin * 2 , _lengthY + margin * 2) );
	}
}
