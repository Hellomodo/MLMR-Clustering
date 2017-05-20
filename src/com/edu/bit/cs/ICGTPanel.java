package com.edu.bit.cs;
import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

public class ICGTPanel extends JPanel
{
	List<Sample> _sample;
	Map<Integer,Color> _mapColor;

	Color[] _arrayColor = {new Color(255,255,255),
							new Color(255,0,0),
							new Color(0,0,255),
							new Color(0,255,0),
							new Color(0,0,0),
							new Color(255,0,255),
							new Color(0,255,255),
							//new Color(255,255,0),
							new Color(125,125,0),
							new Color(125,0,125),
							new Color(0,125,125),
							new Color(125,0,0),
							new Color(0,125,0),
							new Color(0,0,125),
							new Color(125,125,125),
							new Color(0,125,250),
							new Color(0,250,125),
							new Color(125,0,250),
							new Color(250,0,125),
							new Color(125,250,0),
							new Color(250,125,0),
							};
	private int margin = 20;
	private int _lengthX = 1200;
	private int _lengthY = 800;
	private double _minX, _maxX;
	private double _minY, _maxY;
	private double _spanX, _spanY;
	private int _lengthP = 10;

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

		Iterator itSample = _sample.iterator();
		while(itSample.hasNext())
		{
			Sample sample = (Sample)itSample.next();
			if(! _mapColor.containsKey(sample.getPridict()))
			{
				_mapColor.put( sample.getPridict(), new Color((int)(Math.random()*256),(int)(Math.random()*256),(int)(Math.random()*256)) );
			}
			//g.setColor(_mapColor.get( sample.getPridict() ));
			g.setColor(_arrayColor[sample.getPridict()]);
			Point point = coordTransform( sample.variable(0), sample.variable(1));
			//g.fillOval((int)point.getX(),(int)point.getY(), _lengthP, _lengthP);
			//g.drawString("(" + sample.variable(0)+ "," + sample.variable(1) +","+ sample.getPridict() + ")",(int)point.getX(),(int)point.getY() );
			g.drawChars(new String(sample.getPridict() + "  " + "").toCharArray(),0,3,(int)point.getX(),(int)point.getY());
		}
	}

	public void displayClusters(List<Sample> sample)
	{
		_sample = sample;
		_mapColor = new HashMap<Integer, Color>();

		_minX = Double.MAX_VALUE;
		_maxX = 0;
		_minY =  Double.MAX_VALUE ;
		_maxY = 0;

		for(Sample p : _sample)
		{
			double x = p.variable(0), y = p.variable(1);
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
