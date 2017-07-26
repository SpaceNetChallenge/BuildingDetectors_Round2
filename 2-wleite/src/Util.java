import java.awt.Point;
import java.awt.Polygon;
import java.awt.geom.Area;
import java.awt.geom.Line2D;
import java.awt.geom.PathIterator;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

public class Util {
	private static List<Polygon> findCandidates(int w, int h, double cut, double[][] m1, int buildingsPolyBorder, int buildingsPolyBorder2) {
		int minPts = 4;
		List<Polygon> candidates = new ArrayList<Polygon>();

		double[][] m2 = new double[h][w];
		Integer[] ord = new Integer[w * h];
		int oc = 0;
		for (int y = 1; y < h - 1; y++) {
			double a = Math.min(Math.min(m1[y - 1][0], m1[y][0]), m1[y + 1][0]);
			double b = Math.min(Math.min(m1[y - 1][1], m1[y][1]), m1[y + 1][1]);
			for (int x = 1; x < w - 1; x++) {
				double c = Math.min(Math.min(m1[y - 1][x + 1], m1[y][x + 1]), m1[y + 1][x + 1]);
				m2[y][x] = Math.min(a, Math.min(b, c));
				if (m2[y][x] < cut) m2[y][x] = 0;
				else ord[oc++] = y * w + x;
				a = b;
				b = c;
			}
		}
		Arrays.sort(ord, 0, oc, new Comparator<Integer>() {
			public int compare(Integer a, Integer b) {
				double va = m2[a / w][a % w];
				double vb = m2[b / w][b % w];
				return Double.compare(vb, va);
			}
		});
		int[][] m3 = new int[h][w];
		int id = 0;
		int[] q = new int[w * h];
		for (int k = 0; k < oc; k++) {
			int pk = ord[k];
			int x = pk % w;
			int y = pk / w;
			if (m3[y][x] == 0) {
				int tot = 0;
				id++;
				m3[y][x] = id;
				q[tot++] = y * w + x;
				int curr = 0;
				while (curr < tot) {
					int cq = q[curr++];
					int cx = cq % w;
					int cy = cq / w;
					for (int i = 0; i < 4; i++) {
						int nx = i == 0 ? cx + 1 : i == 1 ? cx - 1 : cx;
						int ny = i == 2 ? cy + 1 : i == 3 ? cy - 1 : cy;
						if (m2[ny][nx] > 0 && m3[ny][nx] == 0) {
							m3[ny][nx] = id;
							q[tot++] = ny * w + nx;
						}
					}
				}
				if (tot > minPts) {
					List<Point> pts = new ArrayList<Point>();
					for (int j = 0; j < tot; j++) {
						int cq = q[j];
						int cx = cq % w;
						int cy = cq / w;
						int eq = 0;
						for (int i = 0; i < 4; i++) {
							int nx = i == 0 ? cx + 1 : i == 1 ? cx - 1 : cx;
							int ny = i == 2 ? cy + 1 : i == 3 ? cy - 1 : cy;
							if (m3[ny][nx] == id) eq++;
						}
						if (eq == 4) continue;
						pts.add(new Point(cx, Math.min(h, cy + buildingsPolyBorder + 1)));
						pts.add(new Point(cx, Math.max(0, cy - buildingsPolyBorder)));
						pts.add(new Point(Math.min(w, cx + buildingsPolyBorder + 1), cy));
						pts.add(new Point(Math.max(0, cx - buildingsPolyBorder), cy));
						pts.add(new Point(Math.min(w, cx + buildingsPolyBorder2 + 1), Math.min(h, cy + buildingsPolyBorder2 + 1)));
						pts.add(new Point(Math.max(0, cx - buildingsPolyBorder2), Math.max(0, cy - buildingsPolyBorder2)));
						pts.add(new Point(Math.min(w, cx + buildingsPolyBorder + 1), Math.max(0, cy - buildingsPolyBorder2)));
						pts.add(new Point(Math.max(0, cx - buildingsPolyBorder2), Math.min(h, cy + buildingsPolyBorder2 + 1)));
					}
					List<Point> hull = Util.convexHull(pts);
					if (hull != null) {
						int[] xp = new int[hull.size()];
						int[] yp = new int[hull.size()];
						for (int i = 0; i < xp.length; i++) {
							Point p = hull.get(i);
							xp[i] = p.x;
							yp[i] = p.y;
						}
						Polygon polygon = new Polygon(xp, yp, xp.length);
						candidates.add(polygon);
					}
				}
			}
		}
		return candidates;
	}

	public static List<Polygon> findBuildings(double[][] buildingValue, double[][] borderValue, double buildingsCut, double borderWeight, int buildingsPolyBorder, int buildingsPolyBorder2) {
		int w = borderValue[0].length;
		int h = borderValue.length;
		double[][] m1 = new double[h][w];
		for (int y = 1; y < h - 1; y++) {
			double a = 0;
			double b = 0;
			for (int ay = y - 1; ay <= y + 1; ay++) {
				a += buildingValue[ay][0] - borderWeight * borderValue[ay][0];
				b += buildingValue[ay][1] - borderWeight * borderValue[ay][1];
			}
			for (int x = 1; x < w - 1; x++) {
				double c = 0;
				for (int ay = y - 1; ay <= y + 1; ay++) {
					c += buildingValue[ay][x + 1] - borderWeight * borderValue[ay][x + 1]; //FIXED
				}
				m1[y][x] = (a + b + c + (buildingValue[y][x] - borderWeight * borderValue[y][x])) / 10;
				a = b;
				b = c;
			}
		}
		return findCandidates(w, h, buildingsCut, m1, buildingsPolyBorder, buildingsPolyBorder2);
	}

	public static void splitImages(Map<String, List<Building>> buildingsPerImage, int[] a, int idx) {
		try {
			System.err.println("Spliting Images");
			long t = System.currentTimeMillis();
			List<String> l = new ArrayList<String>(buildingsPerImage.keySet());
			SplittableRandom rnd = new SplittableRandom(14121974);
			for (int i = 0; i < l.size() * l.size(); i++) {
				int p1 = rnd.nextInt(l.size());
				int p2 = rnd.nextInt(l.size());
				Collections.swap(l, p1, p2);
			}
			int p1 = 0;
			for (int i = 0; i < idx; i++) {
				p1 += a[i];
			}
			int p2 = p1 + a[idx];
			p1 = p1 * l.size() / 100;
			p2 = p2 * l.size() / 100;
			if (p2 > l.size()) p2 = l.size();
			buildingsPerImage.keySet().retainAll(l.subList(p1, p2));
			System.err.println("\t         Images: " + buildingsPerImage.size());
			System.err.println("\t   Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-11);
		}
	}

	public static Map<String, List<Building>> readBuildingsCsv(File trainingCsv) {
		Map<String, List<Building>> buildingsPerImage = new HashMap<String, List<Building>>();
		try {
			System.err.println("Reading Buildings CSV");
			long t = System.currentTimeMillis();
			int numLines = 0;
			int numBuildings = 0;
			int numPolys = 0;
			BufferedReader in = new BufferedReader(new FileReader(trainingCsv));
			String line = in.readLine();
			String s1 = "POLYGON ((";
			String s2 = "))";
			while ((line = in.readLine()) != null) {
				numLines++;
				List<String> cols = parseCols(line);
				String imageId = cols.get(0);
				if (imageId.toLowerCase().startsWith("pan_")) imageId = imageId.substring(4);
				List<Building> buildings = buildingsPerImage.get(imageId);
				if (buildings == null) buildingsPerImage.put(imageId, buildings = new ArrayList<Building>());
				int buildingId = Integer.parseInt(cols.get(1));
				if (buildingId == -1) continue;
				Building building = new Building(buildingId);
				buildings.add(building);

				String s = cols.get(2);
				int p1 = s.indexOf(s1);
				boolean first = true;
				while (p1 >= 0) {
					int p2 = s.indexOf(s2, p1 + s1.length());
					String[] pts = s.substring(p1 + s1.length(), p2).split("\\),\\(");
					for (int i = 0; i < pts.length; i++) {
						String[] pt = pts[i].split(",");
						int[] x = new int[pt.length];
						int[] y = new int[pt.length];
						for (int j = 0; j < x.length; j++) {
							String[] coord = pt[j].split(" ");
							x[j] = (int) Math.round(Double.parseDouble(coord[0]));
							y[j] = (int) Math.round(Double.parseDouble(coord[1]));
						}
						Polygon poly = new Polygon(x, y, x.length);
						numPolys++;
						if (first) building.in.add(poly);
						else building.out.add(poly);
					}
					first = false;
					p1 = p2 + s2.length();
					p1 = s.indexOf(s1, p1);
				}
				numBuildings++;
			}
			in.close();
			System.err.println("\t    Lines Read: " + numLines);
			System.err.println("\t        Images: " + buildingsPerImage.size());
			System.err.println("\t     Buildings: " + numBuildings);
			System.err.println("\t      Polygons: " + numPolys);
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		return buildingsPerImage;
	}

	public static double[][][] evalImage(MultiChannelImage mci, RandomForestPredictor buildingPredictor, RandomForestPredictor borderPredictor) {
		try {
			int w = mci.width;
			int h = mci.height;
			double[][] buildingValues = new double[h][w];
			double[][] borderValues = new double[h][w];
			for (int y = 0; y < h; y++) {
				double[] buy = buildingValues[y];
				double[] boy = borderValues[y];
				for (int x = 0; x < w; x++) {
					float[] features = BuildingFeatureExtractor.getFeatures(mci, x, y, w, h);
					buy[x] = buildingPredictor.predict(features);
					boy[x] = borderPredictor.predict(features);
				}
			}
			return new double[][][] { buildingValues, borderValues };
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-14);
		}
		return null;
	}

	private static double triangleArea(int p1x, int p1y, int p2x, int p2y, int p3x, int p3y) {
		return Math.abs(p1x * p2y + p1y * p3x + p2x * p3y - p3x * p2y - p1y * p2x - p1x * p3y) / 2;
	}

	public static double getArea(Polygon a) {
		double ret = 0;
		for (int i = 1; i < a.npoints - 1; i++) {
			ret += triangleArea(a.xpoints[0], a.ypoints[0], a.xpoints[i], a.ypoints[i], a.xpoints[i + 1], a.ypoints[i + 1]);
		}
		return ret;
	}

	public static List<Point> convexHull(List<Point> p) {
		if (p == null) return null;
		if (p.size() < 3) return null;
		Point base = p.get(0);
		for (Point curr : p) {
			if (curr.y < base.y || (curr.y == base.y && curr.x > base.x)) {
				base = curr;
			}
		}
		List<Point> hull = new ArrayList<Point>();
		hull.add(base);
		double prevAng = -1;
		while (true) {
			Point next = null;
			double ang = Math.PI * 3;
			Point last = hull.get(hull.size() - 1);
			for (Point curr : p) {
				if (curr.equals(last)) continue;
				double ca = angle(last, curr);
				if (ca <= ang && ca >= prevAng) {
					ang = ca;
					next = curr;
				}
			}
			prevAng = ang;
			if (next == null || next.equals(base)) break;
			hull.add(next);
		}
		return hull;
	}

	public static double areaVal(Area shape) {
		PathIterator i = shape.getPathIterator(null);
		double a = 0.0;
		double[] coords = new double[6];
		double startX = Double.NaN, startY = Double.NaN;
		Line2D segment = new Line2D.Double(Double.NaN, Double.NaN, Double.NaN, Double.NaN);
		while (!i.isDone()) {
			int segType = i.currentSegment(coords);
			double x = coords[0], y = coords[1];
			switch (segType) {
			case PathIterator.SEG_CLOSE:
				segment.setLine(segment.getX2(), segment.getY2(), startX, startY);
				a += area(segment);
				startX = startY = Double.NaN;
				segment.setLine(Double.NaN, Double.NaN, Double.NaN, Double.NaN);
				break;
			case PathIterator.SEG_LINETO:
				segment.setLine(segment.getX2(), segment.getY2(), x, y);
				a += area(segment);
				break;
			case PathIterator.SEG_MOVETO:
				startX = x;
				startY = y;
				segment.setLine(Double.NaN, Double.NaN, x, y);
				break;
			}
			i.next();
		}
		if (Double.isNaN(a)) {
			throw new IllegalArgumentException("PathIterator contains an open path");
		} else {
			return 0.5 * Math.abs(a);
		}
	}

	private static double area(Line2D seg) {
		return seg.getX1() * seg.getY2() - seg.getX2() * seg.getY1();
	}

	private static double angle(Point pc, Point p) {
		double ang = Math.atan2(p.y - pc.y, p.x - pc.x);
		if (ang < 0) ang += Math.PI * 2;
		return ang;
	}

	private static List<String> parseCols(String line) {
		StringBuilder sb = new StringBuilder();
		List<String> cols = new ArrayList<String>();
		boolean inside = false;
		for (int i = 0; i < line.length(); i++) {
			char c = line.charAt(i);
			if (c == '\"') {
				inside = !inside;
			} else if (c == ',' && !inside) {
				cols.add(sb.toString());
				sb.delete(0, sb.length());
			} else {
				sb.append(c);
			}
		}
		cols.add(sb.toString());
		return cols;
	}
}