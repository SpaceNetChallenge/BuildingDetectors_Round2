/*
 * Building Detector Visualizer and Offline Tester
 * by walrus71
 * 
 * Version history:
 * ================
 * 2.0 (2017.03.10)
 * 		- Version at second contest launch
 * 1.4 (2017.03.10)
 * 		- Params reading made more user friendly
 * 1.3 (2017.03.09)
 * 		- Supports multiple truth files
 * 		- Scoring changed to average of f-scores over cities
 * 1.2 (2017.03.01)
 * 		- Supports new data format for 2nd contest
 * 		- Improved drawing speed
 * 		- Removed RunMode
 * 1.1 (2016.11.16)
 *      - Version at first contest launch
 *      - Minimum building area check added
 *      - Other small fixes
 * 1.0 (2016.11.03)
 *      - First public version
 */
package visualizer;

import static visualizer.Utils.f;
import static visualizer.Utils.f6;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.font.FontRenderContext;
import java.awt.geom.AffineTransform;
import java.awt.geom.Area;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.PathIterator;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

public class BuildingVisualizer implements ActionListener, ItemListener, MouseListener {

	private boolean hasGui = true;
	private String[] imageDirs;
	private String[] imageIds;
	private Map<String, String> idToDir; // which data folder this image comes from
	private String currentImageId;
	private String[] truthPaths;
	private String solutionPath;
	private Map<String, Polygon[]> idToTruthPolygons;
	private Map<String, Polygon[]> idToSolutionPolygons;
	private double iouThreshold = 0.5;
	private static final double MIN_AREA = 20;

	private double scale; // data size / screen size (for 3-band images)
	private double x0 = 0, y0 = 0; // x0, y0: TopLeft corner of data is shown here (in screen space, applies to all views)
	private double ratio38; // scaling factor between 3-band and 8-band images

	private JFrame frame;
	private JPanel viewPanel, controlsPanel;
	private JCheckBox showTruthCb, showSolutionCb, showIouCb;
	private JComboBox<String> viewSelectorComboBox;
	private JComboBox<String> imageSelectorComboBox;
	private JTextArea logArea;
	private MapView mapView;
	private Font font = new Font("SansSerif", Font.BOLD, 14);

	private String bandTripletPath;
	private List<BandTriplet> bandTriplets;
	private BandTriplet currentBandTriplet;

	private Color textColor = Color.black;
	private Color tpBorderSolutionColor = new Color(255, 255, 255, 200);
	private Color tpFillSolutionColor = new Color(255, 255, 255, 50);
	private Color tpBorderTruthColor = new Color(255, 255, 255, 200);
	private Color tpFillTruthColor = new Color(255, 255, 255, 10);
	private Color fpBorderColor = new Color(255, 255, 0, 255);
	private Color fpFillColor = new Color(255, 255, 0, 100);
	private Color fnBorderColor = new Color(0, 255, 255, 255);
	private Color fnFillColor = new Color(0, 155, 255, 100);

	private void run() {
		idToSolutionPolygons = load(new String[] {solutionPath}, false, null);
		idToTruthPolygons = load(truthPaths, true, idToSolutionPolygons);
		imageIds = collectImageIds(idToSolutionPolygons);
		String[] cities = collectCities();

		if (idToSolutionPolygons.isEmpty() || idToTruthPolygons.isEmpty()) {
			// can't score, just output ids
			log("Nothing to score");
			for (String id : imageIds) {
				log(id);
			}
		} else {
			Map<String, Metrics> cityToScore = new HashMap<>();
			for (String c : cities)
				cityToScore.put(c, new Metrics());

			String detailsMarker = "Details:";
			log(detailsMarker);
			for (String id : imageIds) {
				Metrics result = score(id);
				if (result != null) {
					log(id + "\n" + "  TP:\t" + result.tp + "\n" + "  FP:\t" + result.fp + "\n" + "  FN:\t" + result.fn + "\n");

					String city = idToCity(id);
					Metrics m = cityToScore.get(city);
					m.tp += result.tp;
					m.fp += result.fp;
					m.fn += result.fn;
				} else {
					log(id + "\n  - not scored");
				}
			}

			double fSum = 0;
			int cityCnt = 0;
			String result = "";
			String cb = "";
			for (String c : cities) {
				Metrics m = cityToScore.get(c);
				m.calculate();
				fSum += m.fScore;
				cityCnt++;
				result += "\n" + c + ":\n" + "         TP:\t" + m.tp + "\n" + "         FP:\t" + m.fp + "\n" + "         FN:\t" + m.fn + "\n" + "  Precision:\t" + f6(m.precision) + "\n" + "     Recall:\t" + f6(m.recall) + "\n" + "    F-score:\t" + f6(m.fScore);
				cb += m.tp + "\t" + m.fp + "\t" + m.fn + "\t" + f6(m.precision) + "\t" + f6(m.recall) + "\t" + f6(m.fScore) + "\t";
			}
			gcb += cb + "\t" + imageIds.length + "\n";

			if (fSum > 0) {
				double f = fSum / cityCnt;
				result = "\nOverall F-score : " + f6(f) + "\n" + result;
			} else {
				result = "\nOverall F-score : 0\n\n";
			}

			if (hasGui) { // display final result at the top
				String allText = logArea.getText();
				int pos = allText.indexOf(detailsMarker);
				String s1 = allText.substring(0, pos);
				String s2 = allText.substring(pos);
				allText = s1 + result + "\n\n" + s2;
				logArea.setText(allText);
				logArea.setCaretPosition(0);
				System.out.println(result);
			} else {
				log(result);
			}
		} // anything to score

		// the rest is for UI, not needed for scoring
		if (!hasGui) return;

		DefaultComboBoxModel<String> cbm = new DefaultComboBoxModel<>(imageIds);
		imageSelectorComboBox.setModel(cbm);
		imageSelectorComboBox.setSelectedIndex(0);
		imageSelectorComboBox.addItemListener(this);

		currentImageId = imageIds[0];
		loadMap();
		scale = (double) currentBandTriplet.mapData.W / mapView.getWidth();
		repaintMap();
	}

	private Metrics score(String id) {
		Metrics ret = new Metrics();
		Polygon[] truthPolygons = idToTruthPolygons.get(id);
		Polygon[] solutionPolygons = idToSolutionPolygons.get(id);
		if (truthPolygons == null) truthPolygons = new Polygon[0];
		if (solutionPolygons == null) solutionPolygons = new Polygon[0];
		if (truthPolygons.length == 0 && solutionPolygons.length == 0) {
			return null;
		}
		int tp = 0;
		int fp = 0;
		int fn = 0;
		for (Polygon sP : solutionPolygons) {
			Polygon bestMatchingT = null;
			double maxScore = 0;
			for (Polygon tP : truthPolygons) {
				if (tP.match == Match.TP) continue; // matched already
				if (sP.minx > tP.maxx || sP.maxx < tP.minx) continue;
				if (sP.miny > tP.maxy || sP.maxy < tP.miny) continue;
				Area shape = new Area(sP.getShape());
				shape.intersect(tP.getShape());
				double overlap = Math.abs(area(shape));
				double score = overlap / (sP.area + tP.area - overlap);
				if (score > maxScore) {
					maxScore = score;
					bestMatchingT = tP;
				}

			}
			sP.iouScore = maxScore;
			if (maxScore > iouThreshold) {
				tp++;
				sP.match = Match.TP;
				bestMatchingT.match = Match.TP;
			} else {
				fp++;
				sP.match = Match.FP;
			}
		}
		for (Polygon tP : truthPolygons) {
			if (tP.match == Match.NOTHING) {
				fn++;
				tP.match = Match.FN;
			}
		}
		ret.tp = tp;
		ret.fp = fp;
		ret.fn = fn;

		return ret;
	}

	// based on http://stackoverflow.com/questions/2263272/how-to-calculate-the-area-of-a-java-awt-geom-area
	private double area(Area shape) {
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

	private double area(Line2D seg) {
		return seg.getX1() * seg.getY2() - seg.getX2() * seg.getY1();
	}

	private Map<String, Polygon[]> load(String[] paths, boolean truth, Map<String, Polygon[]> ids) {
		String what = truth ? "truth file" : "your solution";
		if (paths == null || paths.length == 0 || paths[0] == null) {
			log("  Path for " + what + " not set, nothing loaded.");
			return new HashMap<>();
		}
		log(" - Reading " + what + " from " + Arrays.toString(paths) + " ...");

		Map<String, List<Polygon>> idToList = new HashMap<>();
		for (String path : paths) {
			String line = null;
			int lineNo = 0;
			try {
				LineNumberReader lnr = new LineNumberReader(new FileReader(path));
				while (true) {
					line = lnr.readLine();
					lineNo++;
					if (line == null) break;
					line = line.trim();
					if (line.isEmpty() || line.startsWith("#") || line.toLowerCase().startsWith("imageid")) continue;
					// ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo | confidence
					// PAN_AOI_5_Khartoum_img1,1,"POLYGON ((124 364 0,...,124 364 0))","POLYGON ((-43 -22 0,...,-43 -22 0))"
					// - or
					// PAN_AOI_5_Khartoum_img1,1,"POLYGON ((124 364 0,...,124 364 0))",0.9
					// - or
					// imgid,-1,POLYGON EMPTY
					// - or
					// imgid,-1,anything

					int pos1 = line.indexOf(",");
					String imageId = line.substring(0, pos1);
					if (imageId.toLowerCase().startsWith("pan_")) imageId = imageId.substring(4);//WLAD

					if (ids != null && !ids.containsKey(imageId)) continue;

					int pos2 = line.indexOf(",", pos1 + 1);
					String buildingId = line.substring(pos1 + 1, pos2);

					List<Polygon> pList = idToList.get(imageId);
					if (pList == null) {
						pList = new Vector<>();
						idToList.put(imageId, pList);
					}

					boolean empty = line.contains("POLYGON EMPTY");
					if (!empty && buildingId.equals("-1")) {
						empty = true;
					}

					if (!empty) {
						pos1 = line.indexOf("((");
						if (pos1 != -1) {
							pos2 = line.indexOf("))", pos1);
							String pString = line.substring(pos1, pos2 + 2);
							Polygon p = new Polygon(pString);
							if (p.area <= 0) {
								if (!truth) {
									log("Warning: building area <= 0");
									log("Line #" + lineNo + ": " + line);
								}
								continue;
							}
							if (p.area < MIN_AREA) {
								continue;
							}
							String confS = line.substring(pos2 + 4);
							if (!truth) {
								p.confidence = Double.parseDouble(confS);
							}

							pList.add(p);
						}
					}
				}
				lnr.close();
			} catch (Exception e) {
				log("Error reading building polygons");
				log("Line #" + lineNo + ": " + line);
				e.printStackTrace();
				System.exit(0);
			}
		} // for paths

		Map<String, Polygon[]> ret = new HashMap<>();
		for (String id : idToList.keySet()) {
			List<Polygon> pList = idToList.get(id);
			Polygon[] pArr = pList.toArray(new Polygon[0]);
			Arrays.sort(pArr);
			ret.put(id, pArr);
		}
		return ret;
	}

	private void loadMap() {
		String baseDir = idToDir.get(currentImageId);
		File dir, f;
		int w3 = 0;
		int[][] rs = null;
		int[][] gs = null;
		int[][] bs = null;

		// load 3-band file from imageDir/RGB-PanSharpen
		dir = new File(baseDir, "RGB-PanSharpen");
		f = new File(dir, "RGB-PanSharpen_" + currentImageId + ".tif");
		if (!f.exists()) {
			log("Can't find image file: " + f.getAbsolutePath());
			return;
		}
		try {
			BufferedImage img = ImageIO.read(f);
			Raster raster = img.getRaster();
			int w = img.getWidth();
			int h = img.getHeight();
			w3 = w;
			rs = new int[w][h];
			gs = new int[w][h];
			bs = new int[w][h];
			int[][] arrs = new int[3][w * h];
			int cnt = 0;
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++) {
					int[] samples = raster.getPixel(i, j, new int[3]);
					for (int b = 0; b < 3; b++) {
						if (samples[b] < 0) samples[b] += 65536; // stored as short
						arrs[b][cnt] = Math.max(arrs[b][cnt], samples[b]);
					}
					cnt++;
					rs[i][j] = samples[0];
					gs[i][j] = samples[1];
					bs[i][j] = samples[2];
				}
			for (int b = 0; b < 3; b++) {
				Arrays.sort(arrs[b]);
			}
			int non0 = 0;
			while (arrs[0][non0] == 0)
				non0++;
			int len = arrs[0].length - non0;
			double[] maxs = new double[3];
			double[] mins = new double[3];
			for (int b = 0; b < 3; b++) {
				maxs[b] = arrs[b][non0 + (int) (0.99 * len)];
				mins[b] = arrs[b][non0 + (int) (0.01 * len)];
			}
			MapData md = new MapData(w, h);
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++) {
					int r = eq(rs[i][j], mins[0], maxs[0]);
					int g = eq(gs[i][j], mins[1], maxs[1]);
					int b = eq(bs[i][j], mins[2], maxs[2]);
					md.pixels[i][j] = toRGB(r, g, b);
				}

			bandTriplets.get(0).mapData = md;
		} catch (Exception e) {
			log("Error reading image from " + f.getAbsolutePath());
			e.printStackTrace();
		}

		// load 1-band grayscale file from imageDir/PAN
		dir = new File(baseDir, "PAN");
		f = new File(dir, "PAN_" + currentImageId + ".tif");
		if (!f.exists()) {
			log("Can't find image file: " + f.getAbsolutePath());
			return;
		}
		try {
			BufferedImage img = ImageIO.read(f);
			Raster raster = img.getRaster();
			int w = img.getWidth();
			int h = img.getHeight();
			int[] arr = new int[w * h];
			int cnt = 0;
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++) {
					int sample = raster.getSample(i, j, 0);
					if (sample < 0) sample += 65536; // stored as short
					arr[cnt] = Math.max(arr[cnt], sample);
					cnt++;
					rs[i][j] = sample;
					gs[i][j] = sample;
					bs[i][j] = sample;
				}
			Arrays.sort(arr);
			int non0 = 0;
			while (arr[non0] == 0)
				non0++;
			int len = arr.length - non0;
			double min = arr[non0 + (int) (0.01 * len)];
			double max = arr[non0 + (int) (0.99 * len)];
			MapData md = new MapData(w, h);
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++) {
					int c = eq(rs[i][j], min, max);
					md.pixels[i][j] = toRGB(c, c, c);
				}

			bandTriplets.get(1).mapData = md;
		} catch (Exception e) {
			log("Error reading image from " + f.getAbsolutePath());
			e.printStackTrace();
		}

		// load 8-band file from imageDir/MUL into 8 arrays first
		dir = new File(baseDir, "MUL");
		f = new File(dir, "MUL_" + currentImageId + ".tif");
		if (!f.exists()) {
			log("Can't find image file: " + f.getAbsolutePath());
			return;
		}
		try {
			BufferedImage img = ImageIO.read(f);
			Raster raster = img.getRaster();
			int w = img.getWidth();
			int h = img.getHeight();
			ratio38 = (double) w3 / w;
			double[][][] bandData = new double[8][w][h];
			int max = 0;
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++) {
					int[] samples = raster.getPixel(i, j, new int[8]);
					for (int b = 0; b < 8; b++) {
						int v = samples[b];
						if (v < 0) v += 65536; // stored as short
						max = Math.max(max, v);
						bandData[b][i][j] = v;
					}
				}
			if (max > 0) {
				for (int b = 0; b < 8; b++)
					for (int i = 0; i < w; i++)
						for (int j = 0; j < h; j++)
							bandData[b][i][j] /= max;
			}

			// create all needed combinations
			for (BandTriplet bt : bandTriplets) {
				if (bt.is3band) continue;
				MapData md = new MapData(w, h);
				if (max > 0) {
					for (int i = 0; i < w; i++)
						for (int j = 0; j < h; j++) {
							int r = (int) (255 * bandData[bt.bands[0] - 1][i][j]);
							int g = (int) (255 * bandData[bt.bands[1] - 1][i][j]);
							int b = (int) (255 * bandData[bt.bands[2] - 1][i][j]);
							md.pixels[i][j] = toRGB(r, g, b);
						}
				}
				bt.mapData = md;
			}
		} catch (Exception e) {
			log("Error reading image from " + f.getAbsolutePath());
			e.printStackTrace();
		}
	}

	private int eq(int c, double min, double max) {
		int v = (int) (255 * (c - min) / (max - min));
		if (v < 0) return 0;
		if (v > 255) return 255;
		return v;
	}

	private int toRGB(int r, int g, int b) {
		return (r << 16) | (g << 8) | b;
	}

	private String[] collectImageIds(Map<String, Polygon[]> tids) {
		Set<String> ids = new HashSet<>();
		idToDir = new HashMap<>();
		for (String dirName : imageDirs) {
			File dir = new File(dirName, "PAN");
			if (!dir.exists() || !dir.isDirectory()) {
				log("Can't find image folder " + dir.getPath());
				continue;
			}
			for (String s : dir.list()) {
				if (!s.endsWith(".tif")) continue;
				s = s.replace(".tif", "");
				s = s.replace("PAN_", "");
				if (!tids.containsKey(s)) continue;
				ids.add(s);
				idToDir.put(s, dirName);
			}
		}
		ids.addAll(idToTruthPolygons.keySet());
		ids.addAll(idToSolutionPolygons.keySet());

		String[] ret = ids.toArray(new String[0]);
		Arrays.sort(ret);
		return ret;
	}

	private String[] collectCities() {
		Set<String> cities = new HashSet<>();
		for (String id : imageIds) {
			String c = idToCity(id);
			cities.add(c);
		}
		String[] arr = cities.toArray(new String[0]);
		Arrays.sort(arr);
		return arr;
	}

	private String idToCity(String id) {
		// AOI_5_Khartoum_img1
		String[] parts = id.split("_");
		return parts[0] + "_" + parts[1] + "_" + parts[2];
	}

	private class P2 {
		public double x;
		public double y;

		public P2(double x, double y) {
			this.x = x;
			this.y = y;
		}

		@Override
		public String toString() {
			return f(x) + ", " + f(y);
		}

		@Override
		public boolean equals(Object o) {
			if (!(o instanceof P2)) return false;
			P2 p = (P2) o;
			double d2 = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
			return d2 < 1e-4;
		}
	}

	private class Metrics {
		public int tp;
		public int fp;
		public int fn;
		public double precision = 0;
		public double recall = 0;
		public double fScore = 0;

		public void calculate() {
			if (tp + fp > 0) precision = (double) tp / (tp + fp);
			if (tp + fn > 0) recall = (double) tp / (tp + fn);
			if (precision + recall > 0) {
				fScore = 2 * precision * recall / (precision + recall);
			}
		}
	}

	private class MapData {
		public int W;
		public int H;
		public int[][] pixels;

		public MapData(int w, int h) {
			W = w;
			H = h;
			pixels = new int[W][H];
		}
	}

	private enum Match {
		NOTHING, TP, FP, FN
	}

	private class Polygon implements Comparable<Polygon> {
		public double confidence;
		public Match match = Match.NOTHING;
		public double minx, miny, maxx, maxy;
		public double iouScore;
		public double area = 0;
		private Area shape;
		public List<Ring> rings = new Vector<>();

		public Polygon(String pString) {
			// ((124 364 0,...,124 364 0),(124 364 0,...,124 364 0))
			pString = pString.replace("),(", "x"); // ring separator
			pString = pString.replace(")", ""); // remove ) and (
			pString = pString.replace("(", "");
			String[] parts = pString.split("x");
			for (String p : parts) {
				Ring r = new Ring(p);
				rings.add(r);
			}
			makeBounds();
			getShape();
		}

		private void makeBounds() {
			minx = Double.MAX_VALUE;
			miny = Double.MAX_VALUE;
			maxx = -Double.MAX_VALUE;
			maxy = -Double.MAX_VALUE;
			for (Ring r : rings) {
				for (P2 p : r.points) {
					minx = Math.min(p.x, minx);
					maxx = Math.max(p.x, maxx);
					miny = Math.min(p.y, miny);
					maxy = Math.max(p.y, maxy);
				}
			}
		}

		public Area getShape() {
			if (shape == null) {
				shape = new Area();
				for (int rI = 0; rI < rings.size(); rI++) {
					Ring r = rings.get(rI);
					Path2D path = new Path2D.Double();
					path.setWindingRule(Path2D.WIND_EVEN_ODD);

					int n = r.points.length;
					path.moveTo(r.points[0].x, r.points[0].y);
					for (int i = 1; i < n; ++i) {
						path.lineTo(r.points[i].x, r.points[i].y);
					}
					path.closePath();
					Area ringArea = new Area(path);
					double a = Math.abs(r.area());
					if (rI == 0) { // first ring is positive
						shape.add(ringArea);
						area += a;
					} else {
						shape.subtract(ringArea);
						area -= a;
					}
				}
			}
			return shape;
		}

		@Override
		public int compareTo(Polygon o) {
			if (this.confidence > o.confidence) return -1;
			if (this.confidence < o.confidence) return 1;
			return 0;
		}

		@Override
		public String toString() {
			return f(minx) + "," + f(miny) + " - " + f(maxx) + "," + f(maxy) + " " + match.toString();
		}
	}

	private class Ring {
		public P2[] points;

		public Ring(String rs) {
			String[] parts = rs.split(",");
			int cnt = parts.length;
			double[] xs = new double[cnt];
			double[] ys = new double[cnt];
			for (int i = 0; i < cnt; i++) {
				String s = parts[i];
				s = s.trim();
				String[] coords = s.split(" ");
				xs[i] = Double.parseDouble(coords[0]);
				ys[i] = Double.parseDouble(coords[1]);
			}

			int n = xs.length;
			points = new P2[n];
			for (int i = 0; i < n; i++)
				points[i] = new P2(xs[i], ys[i]);
			if (n > 1 && !points[0].equals(points[n - 1])) {
				log("Warning: ring not closed: " + rs);
			}
		}

		public double area() {
			// signed area calculated from the points
			double a = 0;
			for (int i = 1; i < points.length; i++) {
				a += (points[i - 1].x + points[i].x) * (points[i - 1].y - points[i].y);
			}
			return a / 2;
		}
	}

	/**************************************************************************************************
	 * THINGS BELOW THIS ARE UI-RELATED, NOT NEEDED FOR SCORING
	 **************************************************************************************************/

	public void setupGUI(int W) {
		if (!hasGui) return;

		loadBandTriplets();

		frame = new JFrame("Building Detector Visualizer");
		int H = W * 2 / 3;
		frame.setSize(W, H);
		frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		Container cp = frame.getContentPane();
		cp.setLayout(new GridBagLayout());

		GridBagConstraints c = new GridBagConstraints();

		c.fill = GridBagConstraints.BOTH;
		c.gridx = 0;
		c.gridy = 0;
		c.weightx = 2;
		c.weighty = 1;
		viewPanel = new JPanel();
		viewPanel.setPreferredSize(new Dimension(H, H));
		cp.add(viewPanel, c);

		c.fill = GridBagConstraints.BOTH;
		c.gridx = 1;
		c.gridy = 0;
		c.weightx = 1;
		controlsPanel = new JPanel();
		cp.add(controlsPanel, c);

		viewPanel.setLayout(new BorderLayout());
		mapView = new MapView();
		viewPanel.add(mapView, BorderLayout.CENTER);

		controlsPanel.setLayout(new GridBagLayout());
		GridBagConstraints c2 = new GridBagConstraints();

		showTruthCb = new JCheckBox("Show truth polygons");
		showTruthCb.setSelected(true);
		showTruthCb.addActionListener(this);
		c2.fill = GridBagConstraints.BOTH;
		c2.gridx = 0;
		c2.gridy = 0;
		c2.weightx = 1;
		controlsPanel.add(showTruthCb, c2);

		showSolutionCb = new JCheckBox("Show solution polygons");
		showSolutionCb.setSelected(true);
		showSolutionCb.addActionListener(this);
		c2.gridy = 1;
		controlsPanel.add(showSolutionCb, c2);

		showIouCb = new JCheckBox("Show IOU scores");
		showIouCb.setSelected(true);
		showIouCb.addActionListener(this);
		c2.gridy = 2;
		controlsPanel.add(showIouCb, c2);

		int b = bandTriplets.size();
		String[] views = new String[b];
		for (int i = 0; i < b; i++)
			views[i] = bandTriplets.get(i).toString();
		viewSelectorComboBox = new JComboBox<>(views);
		viewSelectorComboBox.setSelectedIndex(0);
		viewSelectorComboBox.addItemListener(this);
		c2.gridy = 3;
		controlsPanel.add(viewSelectorComboBox, c2);

		imageSelectorComboBox = new JComboBox<>(new String[] {"..."});
		c2.gridy = 4;
		controlsPanel.add(imageSelectorComboBox, c2);

		JScrollPane sp = new JScrollPane();
		logArea = new JTextArea("", 10, 20);
		logArea.setFont(new Font("Monospaced", Font.PLAIN, 16));
		logArea.addMouseListener(this);
		sp.getViewport().setView(logArea);
		c2.gridy = 5;
		c2.weighty = 10;
		controlsPanel.add(sp, c2);

		frame.setVisible(true);
	}

	private void loadBandTriplets() {
		bandTriplets = new Vector<>();

		BandTriplet bPanSharp = new BandTriplet();
		bPanSharp.is3band = true;
		bPanSharp.name = "RGB Pan-sharpened";
		bandTriplets.add(bPanSharp);
		currentBandTriplet = bPanSharp;

		BandTriplet bPan = new BandTriplet();
		bPan.is3band = true;
		bPan.name = "PAN grayscale";
		bandTriplets.add(bPan);

		String line = null;
		int lineNo = 0;
		try {
			LineNumberReader lnr = new LineNumberReader(new FileReader(bandTripletPath));
			while (true) {
				line = lnr.readLine();
				if (line == null) break;
				lineNo++;
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) continue;
				String[] parts = line.split("\t");
				BandTriplet b = new BandTriplet();
				b.is3band = false;
				b.name = parts[1];
				for (int i = 0; i < 3; i++) {
					b.bands[i] = Integer.parseInt(parts[0].substring(i, i + 1));
				}
				bandTriplets.add(b);
			}
			lnr.close();
		} catch (Exception e) {
			log("Error reading band triplets from " + bandTripletPath);
			log("Line #" + lineNo + " : " + line);
			e.printStackTrace();
			System.exit(0);
		}
	}

	private class BandTriplet {
		public String name;
		public int[] bands = new int[3];
		public boolean is3band;
		public MapData mapData;

		@Override
		public String toString() {
			if (is3band) return name;
			return bands[0] + "," + bands[1] + "," + bands[2] + " : " + name;
		}
	}

	private void repaintMap() {
		if (mapView != null) mapView.repaint();
	}

	@SuppressWarnings("serial")
	private class MapView extends JLabel implements MouseListener, MouseMotionListener, MouseWheelListener {

		private int mouseX;
		private int mouseY;
		private BufferedImage image;
		private int invalidColor = toRGB(50, 150, 200);

		public MapView() {
			super();
			this.addMouseListener(this);
			this.addMouseMotionListener(this);
			this.addMouseWheelListener(this);
		}

		@Override
		public void paint(Graphics gr) {
			if (currentBandTriplet == null || currentBandTriplet.mapData == null) return;
			int W = this.getWidth();
			int H = this.getHeight();
			if (image == null) {
				image = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
			}

			MapData mapData = currentBandTriplet.mapData;

			Graphics2D g2 = (Graphics2D) gr;
			g2.setFont(font);
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
			for (int i = 0; i < W; i++)
				for (int j = 0; j < H; j++) {
					int c = invalidColor;
					int mapI = (int) ((i - x0) * scale);
					int mapJ = (int) ((j - y0) * scale);

					if (mapI >= 0 && mapJ >= 0 && mapI < mapData.W && mapJ < mapData.H) {
						c = mapData.pixels[mapI][mapJ];
					}
					image.setRGB(i, j, c);
				}
			g2.drawImage(image, 0, 0, null);

			if (showTruthCb.isSelected()) {
				Polygon[] truthPolygons = idToTruthPolygons.get(currentImageId);
				if (truthPolygons != null) {
					for (Polygon p : truthPolygons) {
						Color border = p.match == Match.TP ? tpBorderTruthColor : fnBorderColor;
						Color fill = p.match == Match.TP ? tpFillTruthColor : fnFillColor;
						drawPoly(p, g2, border, fill, null);
					}
				}
			}
			if (showSolutionCb.isSelected()) {
				Polygon[] solutionPolygons = idToSolutionPolygons.get(currentImageId);
				if (solutionPolygons != null) {
					for (Polygon p : solutionPolygons) {
						String label = null;
						if (showIouCb.isSelected()) {
							label = f(p.iouScore);
						}
						Color border = p.match == Match.TP ? tpBorderSolutionColor : fpBorderColor;
						Color fill = p.match == Match.TP ? tpFillSolutionColor : fpFillColor;
						drawPoly(p, g2, border, fill, label);
					}
				}
			}
		}

		private void drawPoly(Polygon p, Graphics2D g2, Color border, Color fill, String label) {
			// polygon coordinates are in 3-band space so everything should be scaled if needed
			double r = currentBandTriplet.is3band ? 1 : ratio38;

			double minx = p.minx / r / scale + x0;
			if (minx > this.getWidth()) return;
			double maxx = p.maxx / r / scale + x0;
			if (maxx < 0) return;
			double miny = p.miny / r / scale + y0;
			if (miny > this.getHeight()) return;
			double maxy = p.maxy / r / scale + y0;
			if (maxy < 0) return;

			AffineTransform t = new AffineTransform();
			t.translate(x0, y0);
			t.scale(1 / (r * scale), 1 / (r * scale));
			Area a = p.getShape().createTransformedArea(t);

			g2.setColor(border);
			g2.draw(a);
			g2.setColor(fill);
			g2.fill(a);

			if (label != null) {
				int centerX = (int) ((p.maxx / r + p.minx / r) / 2 / scale + x0);
				int centerY = (int) ((p.maxy / r + p.miny / r) / 2 / scale + y0);
				int w = textWidth(label, g2);
				int h = font.getSize();
				g2.setColor(textColor);
				g2.drawString(label, centerX - w / 2, centerY + h / 2);
			}
		}

		private int textWidth(String text, Graphics2D g) {
			FontRenderContext context = g.getFontRenderContext();
			Rectangle2D r = font.getStringBounds(text, context);
			return (int) r.getWidth();
		}

		@Override
		public void mouseClicked(java.awt.event.MouseEvent e) {
			// nothing
		}

		@Override
		public void mouseReleased(java.awt.event.MouseEvent e) {
			repaintMap();
		}

		@Override
		public void mouseEntered(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
		}

		@Override
		public void mouseExited(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getDefaultCursor());
		}

		@Override
		public void mousePressed(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			mouseX = x;
			mouseY = y;
			repaintMap();
		}

		@Override
		public void mouseDragged(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			x0 += x - mouseX;
			y0 += y - mouseY;
			mouseX = x;
			mouseY = y;
			repaintMap();
		}

		@Override
		public void mouseMoved(java.awt.event.MouseEvent e) {
			// ignore
		}

		@Override
		public void mouseWheelMoved(MouseWheelEvent e) {
			mouseX = e.getX();
			mouseY = e.getY();
			double dataX = (mouseX - x0) * scale;
			double dataY = (mouseY - y0) * scale;

			double change = Math.pow(2, 0.5);
			if (e.getWheelRotation() > 0) scale *= change;
			if (e.getWheelRotation() < 0) scale /= change;

			x0 = mouseX - dataX / scale;
			y0 = mouseY - dataY / scale;

			repaintMap();
		}
	} // class MapView

	@Override
	public void actionPerformed(ActionEvent e) {
		// check boxes clicked
		repaintMap();
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		if (e.getStateChange() == ItemEvent.SELECTED) {
			if (e.getSource() == imageSelectorComboBox) {
				// new image selected
				currentImageId = (String) imageSelectorComboBox.getSelectedItem();
				loadMap();
			} else if (e.getSource() == viewSelectorComboBox) {
				BandTriplet old = currentBandTriplet;
				// new band triplet selected
				int i = viewSelectorComboBox.getSelectedIndex();
				currentBandTriplet = bandTriplets.get(i);
				// 3 -> 8
				if (old.is3band && !currentBandTriplet.is3band) scale /= ratio38;
				// 8 -> 3
				if (!old.is3band && currentBandTriplet.is3band) scale *= ratio38;
			}
			repaintMap();
		}
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		if (e.getSource() != logArea) return;
		try {
			int lineIndex = logArea.getLineOfOffset(logArea.getCaretPosition());
			int start = logArea.getLineStartOffset(lineIndex);
			int end = logArea.getLineEndOffset(lineIndex);
			String line = logArea.getDocument().getText(start, end - start).trim();
			for (int i = 0; i < imageIds.length; i++) {
				if (imageIds[i].equals(line)) {
					currentImageId = imageIds[i];
					imageSelectorComboBox.setSelectedIndex(i);
					loadMap();
					repaintMap();
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public void mousePressed(MouseEvent e) {
	}

	@Override
	public void mouseReleased(MouseEvent e) {
	}

	@Override
	public void mouseEntered(MouseEvent e) {
	}

	@Override
	public void mouseExited(MouseEvent e) {
	}

	private void log(String s) {
		if (logArea != null) logArea.append(s + "\n");
		System.out.println(s);
	}

	private static Color parseColor(String s) {
		String[] parts = s.split(",");
		int r = Integer.parseInt(parts[0]);
		int g = Integer.parseInt(parts[1]);
		int b = Integer.parseInt(parts[2]);
		int a = parts.length > 3 ? Integer.parseInt(parts[3]) : 255;
		return new Color(r, g, b, a);
	}

	private static void exit(String s) {
		System.out.println(s);
		System.exit(1);
	}

	private static String[] parseParamFile(String path) {
		List<String> paramList = new Vector<>();
		try {
			List<String> truthList = new Vector<>();
			List<String> imageDirList = new Vector<>();
			List<String> lines = Utils.readTextLines(path);
			for (String line : lines) {
				if (!line.startsWith("-")) line = "-" + line;
				int pos = line.indexOf("=");
				if (pos == -1) {
					paramList.add(line);
					continue;
				}
				String key = line.substring(0, pos).trim();
				String value = line.substring(pos + 1).trim();
				if (key.equals("-image-dir")) {
					imageDirList.add(value);
				} else if (key.equals("-truth")) {
					truthList.add(value);
				} else {
					paramList.add(key);
					paramList.add(value);
				}
			} // for lines
			if (!truthList.isEmpty()) {
				paramList.add("-truth");
				String p = "";
				for (String s : truthList)
					p += s + SEP;
				p = p.substring(0, p.length() - 1);
				paramList.add(p);
			}
			if (!imageDirList.isEmpty()) {
				paramList.add("-image-dir");
				String p = "";
				for (String s : imageDirList)
					p += s + SEP;
				p = p.substring(0, p.length() - 1);
				paramList.add(p);
			}
		} catch (Exception e) {
			e.printStackTrace();
			exit("Can't parse params file " + path);
		}
		return paramList.toArray(new String[0]);
	}

	private static final String SEP = ";";

	public static void main(String[] args) throws Exception {
		boolean setDefaults = true;
		for (int i = 0; i < args.length; i++) { // to change settings easily from Eclipse
			if (args[i].equals("-no-defaults")) setDefaults = false;
		}

		BuildingVisualizer v = new BuildingVisualizer();
		v.hasGui = true;
		int w = 1500;

		if (setDefaults) {
			v.hasGui = true;
			w = 1500;
			v.truthPaths = null;
			v.solutionPath = null;
			v.imageDirs = null;
			v.bandTripletPath = null;
		} else {
			String params;
			// These are just some default settings for local testing, can be ignored.

			// sample data
			params = "../data/spacenet_sample/params.txt";

			// training data
			//			params = "../data/train/params.txt";

			// test data
			//			params = "../data/test/params.txt";

			// validation
			//			params = "../data/validate/params.txt";

			args = new String[] {"-params",params};
		}

		if (args.length == 2 && args[0].equals("-params")) {
			args = parseParamFile(args[1]);
		}

		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-no-gui")) v.hasGui = false;
			if (args[i].equals("-w")) w = Integer.parseInt(args[i + 1]);
			if (args[i].equals("-iou-threshold")) v.iouThreshold = Double.parseDouble(args[i + 1]);
			if (args[i].equals("-truth")) v.truthPaths = args[i + 1].split(SEP);
			if (args[i].equals("-solution")) v.solutionPath = args[i + 1];
			if (args[i].startsWith("-image-dir")) v.imageDirs = args[i + 1].split(SEP);
			if (args[i].equals("-band-triplets")) v.bandTripletPath = args[i + 1];
			if (args[i].equals("-tp-border-solution")) v.tpBorderSolutionColor = parseColor(args[i + 1]);
			if (args[i].equals("-tp-fill-solution")) v.tpFillSolutionColor = parseColor(args[i + 1]);
			if (args[i].equals("-tp-border-truth")) v.tpBorderTruthColor = parseColor(args[i + 1]);
			if (args[i].equals("-tp-fill-truth")) v.tpFillTruthColor = parseColor(args[i + 1]);
			if (args[i].equals("-fp-border")) v.fpBorderColor = parseColor(args[i + 1]);
			if (args[i].equals("-fp-fill")) v.fpFillColor = parseColor(args[i + 1]);
			if (args[i].equals("-fn-border")) v.fnBorderColor = parseColor(args[i + 1]);
			if (args[i].equals("-fn-fill")) v.fnFillColor = parseColor(args[i + 1]);
		}

		File sp = new File(v.solutionPath);
		if (sp != null && sp.exists() && sp.isDirectory()) {
			File[] files = sp.listFiles();
			if (files != null && files.length > 0) {
				Arrays.sort(files, new Comparator<File>() {
					public int compare(File a, File b) {
						int cmp = Boolean.compare(a.isDirectory(), b.isDirectory());
						if (cmp != 0) return cmp;
						//return Long.compare(b.lastModified(), a.lastModified());
						return a.getName().compareTo(b.getName());
					}
				});
				for (File a : files) {
					if (a.isFile()) {
						v.solutionPath = a.getAbsolutePath();
						System.err.println(">> " + v.solutionPath);
						gcb += a.getName() + "\t";
						v.setupGUI(w);
						v.run();
					}
				}
			}
		}
		/*
				if (v.hasGui && (v.imageDirs == null || v.imageDirs.length == 0)) {
					exit("Image folders not set or empty.");
				}
		
				v.setupGUI(w);
				v.run();
		*/
		System.err.println("DONE");
		System.err.println(gcb);
		try {
			StringSelection stringSelection = new StringSelection(gcb);
			Clipboard clpbrd = Toolkit.getDefaultToolkit().getSystemClipboard();
			clpbrd.setContents(stringSelection, null);
		} catch(Exception e) {
		}
	}

	static String gcb = "";
}
