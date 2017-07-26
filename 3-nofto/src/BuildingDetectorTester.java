import java.awt.Polygon;
import java.awt.geom.Area;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.imageio.ImageIO;

public class BuildingDetectorTester {
	private RandomForestPredictor buildingPredictor, borderPredictor, distPredictor, polyMatchPredictor;
	private final int numThreads = Runtime.getRuntime().availableProcessors();
	private final List<String> answers = new ArrayList<String>();
	private final List<String> testingImages = new ArrayList<String>();

	public static void main(String[] args) {
		//args = new String[]{"c:\\competition1\\spacenet_TestData","c:\\wleite\\model","c:\\sub.csv"};
		/*
		boolean isLocal = true;

		File rfBuilding = new File("out/rfBuilding.dat");
		File rfBorder = new File("out/rfBorder.dat");
		File rfDist = new File("out/rfDist.dat");
		File rfPolyMatch = new File("out/rfPolyMatch.dat");
		
		File folder3band = new File("../competition1/spacenet_TrainData/3band");
		File folder8band = new File("../competition1/spacenet_TrainData/8band");
		if (!isLocal) {
			folder3band = new File("../competition1/spacenet_TestData/3band");
			folder8band = new File("../competition1/spacenet_TestData/8band");
		}
		File answerFile = null;
		for (int i = 1;; i++) {
			answerFile = new File("out/res" + i + ".csv");
			if (!answerFile.exists()) break;
		}
		*/
		/*if (args.length != 3) {
			System.err.println("usage: java BuildingDetectorTester [Path to testing data folder] [Path to model folder] [Output file]");
			return;
		}*/
		boolean isLocal = false;
		
		String dataDir = "data/AOI_3_Paris_Test_public";
	  if(args.length > 0) dataDir = args[0];
	  int position = dataDir.indexOf("AOI_");
	  if(position == -1){
	    System.err.println("");
	    System.err.println("    Input directory does not contain 'AOI_' substring! Exiting without doing anything...");
	    System.exit(1);
	  }
	  String test = dataDir.substring(position + 4, position + 5);
	  String testName = dataDir.substring(position);
	  String outDir = "models/city" + test + "/";
		
	  File folder3band = new File(dataDir + "/RGB-PanSharpen");
		File folder8band = new File(dataDir + "/MUL-PanSharpen");
		
		/*File dataFolder = new File(args[0]);
		File folder3band = new File(dataFolder, "RGB-PanSharpen");
		File folder8band = new File(dataFolder, "MUL-PanSharpen");*/

		//File modelFolder = new File(args[1]);
		
		File rfBuilding = new File(outDir + "rfBuilding.dat");
		File rfBorder = new File(outDir + "rfBorder.dat");
		File rfDist = new File(outDir + "rfDist.dat");
		File rfPolyMatch = new File(outDir + "rfPolyMatch.dat");
		
		/*File rfBuilding = new File(modelFolder, "rfBuilding.dat");
		File rfBorder = new File(modelFolder, "rfBorder.dat");
		File rfDist = new File(modelFolder, "rfDist.dat");
		File rfPolyMatch = new File(modelFolder, "rfPolyMatch.dat");*/
		
	  File answerFile = new File((args.length > 1) ? args[1] : "out" + test + ".csv");

	  System.err.println("");
		System.err.println("   Testing for data '" + testName + "' succesfully started...");
		System.err.println("");
	  
		new BuildingDetectorTester().runTest(rfBuilding, rfBorder, rfDist, rfPolyMatch, folder3band, folder8band, answerFile, isLocal);
	}

	public void runTest(File rfBuilding, File rfBorder, File rfDist, File rfPolyMatch, File folder3band, File folder8band, File answerFile, boolean isLocal) {
		buildingPredictor = loadPredictor(rfBuilding);
		borderPredictor = loadPredictor(rfBorder);
		distPredictor = loadPredictor(rfDist);
		polyMatchPredictor = loadPredictor(rfPolyMatch);

		if (isLocal) {
			File trainingCsv = new File("../competition1/spacenet_TrainData/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv");
			Map<String, List<Building>> buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
			Util.splitImages(buildingsPerImage, new int[] {65,35,0}, 2);
			testingImages.addAll(buildingsPerImage.keySet());

			/*
			Collections.sort(testingImages, new Comparator<String>() {
				public int compare(String a, String b) {
					return Integer.compare(buildingsPerImage.get(b).size(), buildingsPerImage.get(a).size());
				}
			});
			testingImages.subList(100, testingImages.size()).clear();
			*/
		} else {
			File[] files = folder3band.listFiles();
			List<String> ids = new ArrayList<String>();
			String prefix = "RGB-PanSharpen_";
			String suffix = ".tif";
			for (File file : files) {
				if (file.getName().startsWith(prefix) && file.getName().endsWith(suffix)) {
					String id = file.getName().substring(prefix.length(), file.getName().length() - suffix.length());
					ids.add(id);
					//System.err.println(id);
				}
			}
			//System.err.println("===REAL TEST===");
			testingImages.addAll(ids);
		}
		//testingImages.subList(testingImages.size() / 50, testingImages.size()).clear();

		processImages(folder3band, folder8band);
		writeAnswer(answerFile, true/*answers.size() > 1000*/);
	}

	private void processImages(File folder3band, File folder8band) {
		try {
			System.err.println("        Processing Images with " + numThreads + " parallel threads...");
			long t = System.currentTimeMillis();
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				final int start = i;
				threads[i] = new Thread() {
					public void run() {
						for (int j = start; j < testingImages.size(); j += numThreads) {
							String id = testingImages.get(j);
							System.err.print("\r\t" + (j+1) + " / " + testingImages.size() + "        " /*"\t" + id*/);
							processImage(id, folder3band, folder8band);
							// if (start == 0) writeAnswer(answerFile, false);
						}
					}
				};
				threads[i].start();
				threads[i].setPriority(Thread.MIN_PRIORITY);
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
 			System.err.println("");
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void writeAnswer(File answerFile, boolean checkEmpty) {
		try {
			System.err.println("        Writing Answer");
			long t = System.currentTimeMillis();
			Set<String> seen = new HashSet<String>();
			//BufferedWriter out = answerFile.exists() ? new BufferedWriter(new FileWriter(answerFile, true)) : new BufferedWriter(new FileWriter(answerFile));
			BufferedWriter out = new BufferedWriter(new FileWriter(answerFile, true));
			for (int i = 0; i < answers.size(); i++) {
				String line = answers.get(i);
				out.write(line);
				out.newLine();
				int p = line.indexOf(',');
				if (p > 0) seen.add(line.substring(0, p));
			}
			if (checkEmpty) {
				for (String id : testingImages) {
					if (seen.contains(id)) continue;
					out.write(id);
					out.write(",-1,POLYGON EMPTY,1");
					out.newLine();
				}
			}
			out.close();
			System.err.println("\t          File: " + answerFile.getPath());
			System.err.println("\t Lines Written: " + answers.size());
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(String imageId, File folder3band, File folder8band) {
		try {
			File imageFile3band = new File(folder3band, "RGB-PanSharpen_" + imageId + ".tif");
			File imageFile8band = new File(folder8band, "MUL-PanSharpen_" + imageId + ".tif");
			BufferedImage img3band = ImageIO.read(imageFile3band);
			BufferedImage img8band = ImageIO.read(imageFile8band);
			MultiChannelImage mci = new MultiChannelImage(img3band, img8band);
			double[][][] vals = Util.evalImage(mci, buildingPredictor, borderPredictor, distPredictor);
			double[][] buildingValues = vals[0];
			double[][] borderValues = vals[1];
			double[][] distValues = vals[2];

			double minProb = 0.35;
			List<MatchPolygon> candidates = new ArrayList<MatchPolygon>();
			for (int borderShift : PolygonFeatureExtractor.borderShifts) {
				for (double borderWeight : PolygonFeatureExtractor.borderWeights) {
					for (double cut : PolygonFeatureExtractor.buildingsCuts) {
						List<Polygon> polygons = Util.findBuildings(buildingValues, borderValues, distValues, cut, borderWeight, PolygonFeatureExtractor.buildingsPolyBorder + borderShift, PolygonFeatureExtractor.buildingsPolyBorder2 + borderShift);
						for (Polygon polygon : polygons) {
							double prob = polyMatchPredictor.predict(PolygonFeatureExtractor.getFeatures(mci, polygon, buildingValues, borderValues, distValues));
							if (prob > minProb) candidates.add(new MatchPolygon(prob, polygon));
						}
					}
				}
			}

			double maxInter = 0.27;
			Collections.sort(candidates);
			List<Area> areas = new ArrayList<Area>();
			List<Double> aa = new ArrayList<Double>();
			NEXT: for (int i = 0; i < candidates.size(); i++) {
				Polygon pi = candidates.get(i).polygon;
				double pia = Util.getArea(pi);
				Area ai = new Area(pi);
				for (int j = 0; j < i; j++) {
					Polygon pj = candidates.get(j).polygon;
					if (pi.getBounds().getMaxX()<=pj.getBounds().getMinX()) continue;
					if (pi.getBounds().getMaxY()<=pj.getBounds().getMinY()) continue;
					if (pj.getBounds().getMaxX()<=pi.getBounds().getMinX()) continue;
					if (pj.getBounds().getMaxY()<=pi.getBounds().getMinY()) continue;
					Area aj = new Area(areas.get(j));
					aj.intersect(ai);
					double aaj = Util.areaVal(aj);
					double pja = aa.get(j);
					if (aaj > maxInter * pia || aaj > maxInter * pja) {
						candidates.remove(i--);
						continue NEXT;
					}
				}
				areas.add(ai);
				aa.add(pia);
			}

			for (int i = 0; i < candidates.size(); i++) {
				Polygon polygon = candidates.get(i).polygon;
				StringBuilder sb = new StringBuilder();
				sb.append(imageId);
				sb.append(",");
				sb.append(i);
				sb.append(",");
				sb.append("\"POLYGON ((");
				for (int j = 0; j <= polygon.npoints; j++) {
					if (j > 0) sb.append(", ");
					int k1 = j == polygon.npoints ? 0 : j;
					sb.append(polygon.xpoints[k1]);
					sb.append(" ");
					sb.append(polygon.ypoints[k1]);
					sb.append(" ");
					sb.append(0);
				}
				sb.append("))\"");
				sb.append(",");
				sb.append(candidates.size() - i);
				answers.add(sb.toString());
			}

			/////////////////////////////////////////////////////////////////////////////
			/*
			int w = img3band.getWidth();
			int h = img3band.getHeight();
			BufferedImage img2 = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_BGR);
			Graphics2D g = img2.createGraphics();
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.setStroke(new BasicStroke(0.5f));
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int bb = (int) (255 * borderValues[y][x]);
					int v = (int) (255 * buildingValues[y][x]) + 256 * 256 * bb;
					img2.setRGB(x, y, v);
					img2.setRGB(x + w, y, img3band.getRGB(x, y));
				}
			}
			
			for (int i = 0; i < candidates.size(); i++) {
				g.setColor(Color.yellow);
				g.draw(candidates.get(i).polygon);
			}
			g.dispose();
			new ImgViewer(img2, imageId);
			*/
			/////////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////////
			/*
			BufferedImage img3 = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_BGR);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					img3.setRGB(x, y, mci.edge[y * w + x] * 255 / 1000);
					int y8 = y / 4;
					int x8 = x / 4;
					if (y8 < mci.h8 && x8 < mci.w8) {
						int v = mci.extraBands[0][(y / 4) * mci.w8 + x / 4] * 255 / 3000;
						if (v > 255) v = 255;
						img3.setRGB(x, y, v);
					}
					img3.setRGB(x + w, y, img3band.getRGB(x, y));
				}
			}
			g.dispose();
			new ImgViewer(img3, imageId);
			*/
			/////////////////////////////////////////////////////////////////////////////
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private RandomForestPredictor loadPredictor(File rfFile) {
		try {
			return RandomForestPredictor.load(rfFile, -1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}