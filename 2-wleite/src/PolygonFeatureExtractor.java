import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PolygonFeatureExtractor {
	public static final double[] buildingsCuts = new double[] {0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.30,0.33,0.36,0.39,0.42,0.45,0.48,0.51,0.54,0.57,0.60,0.63,0.66,0.69,0.72,0.75,0.78}; 
	public static final double[] borderWeights = new double[] {1, 0.25, -0.5};
	public static final int[] borderShifts = new int[] {0, -2, -4};
	public static final int buildingsPolyBorder = 5;
	public static final int buildingsPolyBorder2 = 4;
	public static final int numFeatures = 209;
	private final int[][] dist;
	private final List<List<Double>> values8band = new ArrayList<List<Double>>(), values3band = new ArrayList<List<Double>>();
	
	public PolygonFeatureExtractor(int w, int h){
		dist = new int[h][w];
		for (int i = 0; i < 7 * 6; i++) {
			values3band.add(new ArrayList<Double>());
		}
		for (int i = 0; i < 16 * 6; i++) {
			values8band.add(new ArrayList<Double>());
		}
	}
	
	public float[] getFeatures(MultiChannelImage mci, Polygon polygon, double[][] buildingValues, double[][] borderValues) {
		float[] ret = new float[numFeatures];
		int k = 0;
		double area = Util.getArea(polygon);
		double[] r = rectLen(polygon);
		ret[k++] = (float) area;
		ret[k++] = (float) r[0];
		ret[k++] = (float) r[1];
		ret[k++] = (float) (r[1] == 0 ? 0 : r[0] / r[1]);
		ret[k++] = (float) (area == 0 ? 0 : r[0] * r[1] / area);

		Rectangle rc = polygon.getBounds();
		int cnt = 0;
		double buildingSum = 0;
		double borderSum = 0;
		final int w = borderValues[0].length;
		final int h = borderValues.length;
		boolean[][] inside = new boolean[h][w];
		for (int y = rc.y; y <= rc.y + rc.height; y++) {
			int x0 = rc.x;
			for (int x = rc.x; x <= rc.x + rc.width; x++) {
				if (polygon.contains(x, y)) {
					x0 = x;
					break;
				}
			}
			int x1 = rc.x + rc.width;
			for (int x = x1; x >= x0; x--) {
				if (polygon.contains(x, y)) {
					x1 = x;
					break;
				}
			}
			for (int x = x0; x <= x1; x++) {
				cnt++; 
				if (x >= 0 && y >= 0 && x < w && y < h) {
					inside[y][x] = true;
					buildingSum += buildingValues[y][x];
					borderSum += borderValues[y][x];
				}
			}
		}
		ret[k++] = (float) borderSum;
		ret[k++] = (float) buildingSum;
		ret[k++] = cnt == 0 ? -1 : (float) (borderSum / cnt);
		ret[k++] = cnt == 0 ? -1 : (float) (buildingSum / cnt);
		ret[k++] = buildingSum == 0 ? -1 : (float) (borderSum / buildingSum);
		ret[k++] = (float) (buildingSum - borderSum);

		final int maxDist = 8;
		final int halfDist = maxDist / 2;
		for (int y = 0; y < h; y++) {
			Arrays.fill(dist[y], 1000);
		}
		int y0 = Math.max(0, rc.y - maxDist);
		int y1 = Math.min(h - 1, rc.y + rc.height + maxDist);
		int x0 = Math.max(0, rc.x - maxDist);
		int x1 = Math.min(w - 1, rc.x + rc.width + maxDist);
		int[] queue = new int[(y1 - y0 + 1) * (x1 - x0 + 1) * 4];
		int tot = 0;
		for (int y = y0; y <= y1; y++) {
			boolean[] iy = inside[y];
			for (int x = x0; x <= x1; x++) {
				boolean v = iy[x];
				boolean b = false;
				if (x > 0 && v != iy[x - 1]) b = true;
				else if (x < w - 1 && v != iy[x + 1]) b = true;
				else if (y > 0 && v != inside[y - 1][x]) b = true;
				else if (y < h - 1 && v != inside[y + 1][x]) b = true;
				if (b) {
					queue[tot++] = y * w + x;
					dist[y][x] = 0;
				}
			}
		}
		int curr = 0;
		while (curr < tot) {
			int p = queue[curr++];
			int x = p % w;
			int y = p / w;
			int nd = dist[y][x] + 1;
			if (nd > maxDist) continue;
			for (int i = 0; i < 4; i++) {
				int nx = i == 0 ? x + 1 : i == 1 ? x - 1 : x;
				if (nx < 0 || nx >= w) continue;
				int ny = i == 2 ? y + 1 : i == 3 ? y - 1 : y;
				if (ny < 0 || ny >= h) continue;
				if (dist[ny][nx] > nd) {
					dist[ny][nx] = nd;
					queue[tot++] = ny * w + nx;
				}
			}
		}
		for (List<Double> l : values3band) {
			l.clear();
		}
		for (List<Double> l : values8band) {
			l.clear();
		}
		for (int y = y0; y <= y1; y++) {
			boolean[] iy = inside[y];
			int[] dy = dist[y];
			double[] buy = buildingValues[y];
			double[] boy = borderValues[y];
			for (int x = x0; x <= x1; x++) {
				boolean v = iy[x];
				int d = dy[x];
				int idx = -1;
				if (d == 0) idx = 0;
				else if (d <= halfDist && v) idx = 1;
				else if (d <= halfDist && !v) idx = 2;
				else if (d <= maxDist && v) idx = 3;
				else if (d <= maxDist && !v) idx = 4;
				else if (d > maxDist && v) idx = 5;
				if (idx >= 0) {
					int p = y * w + x;
					double a = buy[x];
					double b = boy[x];
					int pos = idx * 7;
					values3band.get(pos++).add(a);
					values3band.get(pos++).add(b);
					values3band.get(pos++).add(a - b);
					values3band.get(pos++).add((double) mci.edge[p]);
					values3band.get(pos++).add((double) mci.h[p]);
					values3band.get(pos++).add((double) mci.s[p]);
					values3band.get(pos++).add((double) mci.gray[p]); 
					int x8 = x / 4;
					int y8 = y / 4;
					pos = idx * 16;
					if (x8 < mci.w8 && y8 < mci.h8) {
						p = y8 * mci.w8 + x8;
						for (int i = 0; i < 8; i++) {
							values8band.get(pos++).add((double) mci.extraBands[i][p]);
							values8band.get(pos++).add((double) mci.extraEdges[i][p]);
						}
					}
				}
			}
		}
		int idx = 0;
		for (List<Double> a : values3band) {
			int n = idx % 7 < 3 ? 3 : 2;
			System.arraycopy(stats(a), 0, ret, k, n);
			k += n;
			idx++;
		}
		for (List<Double> a : values8band) {
			ret[k++] = avg(a);
		}
		return ret;
	}

	private static float[] stats(List<Double> l) {
		double sum = 0;
		double sumSquares = 0;
		int cnt = 0;
		for (double p : l) {
			sum += p;
			double pp = p * p;
			sumSquares += pp;
			cnt++;
		}
		float[] ret = new float[3];
		if (cnt > 0) {
			ret[0] = (float) (sum / cnt);
			ret[1] = (float) ((sumSquares - sum * sum / cnt) / cnt);
			ret[2] = (float) sum;
		}
		return ret;
	}

	private static float avg(List<Double> l) {
		double sum = 0;
		for (double p : l) {
			sum += p;
		}
		return l.size() == 0 ? -1 : (float) (sum / l.size());
	}

	private static double[] rectLen(Polygon polygon) {
		Rectangle rc = polygon.getBounds();
		double xc = rc.getCenterX();
		double yc = rc.getCenterY();
		Point2D[] org = new Point2D[polygon.npoints];
		Point2D[] dst = new Point2D[polygon.npoints];
		for (int i = 0; i < polygon.npoints; i++) {
			org[i] = new Point2D.Double(polygon.xpoints[i], polygon.ypoints[i]);
		}
		double minArea = 1e99;
		double[] ret = new double[2];
		for (int a = 0; a < 90; a += 3) {
			AffineTransform.getRotateInstance(Math.toRadians(a), xc, yc).transform(org, 0, dst, 0, org.length);
			Point2D r = dst[0];
			double xmin = r.getX();
			double ymin = r.getY();
			double xmax = xmin;
			double ymax = ymin;
			for (int i = 1; i < polygon.npoints; i++) {
				r = dst[i];
				xmin = Math.min(xmin, r.getX());
				xmax = Math.max(xmax, r.getX());
				ymin = Math.min(ymin, r.getY());
				ymax = Math.max(ymax, r.getY());
			}
			double dx = xmax - xmin;
			double dy = ymax - ymin;
			double currArea = dx * dy;
			if (currArea < minArea) {
				minArea = currArea;
				ret[0] = Math.min(dx, dy);
				ret[1] = Math.max(dx, dy);
			}
		}
		return ret;
	}
}