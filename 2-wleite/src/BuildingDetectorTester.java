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
	private RandomForestPredictor buildingPredictor, borderPredictor, polyMatchPredictor;
	private final int numThreads = SpacenetMain.numThreads;
	private final List<String> answers = new ArrayList<String>();
	private final List<String> testingImages = new ArrayList<String>();
	private static boolean isLocal = false;
	private int minConf;

	public void run(String trainingFolder, String dataSet, int minConf, File answerFile) {
		this.minConf = minConf;
		File trainingCsv = isLocal ? new File(trainingFolder, "summaryData/AOI_" + dataSet + "_Train_Building_Solutions.csv") : null;
		File folderPan = new File(trainingFolder, "PAN");
		File folder3band = new File(trainingFolder, "RGB-PanSharpen");
		File folder8band = new File(trainingFolder, "MUL");
		File rfModel = new File("model", dataSet);
		File rfBuilding = new File(rfModel, "rfBuilding.dat");
		File rfBorder = new File(rfModel, "rfBorder.dat");
		File rfPolyMatch = new File(rfModel, "rfPolyMatch.dat");
		runTest(rfBuilding, rfBorder, rfPolyMatch, folderPan, folder3band, folder8band, answerFile, trainingCsv);
	}

	public void runTest(File rfBuilding, File rfBorder, File rfPolyMatch, File folderPan, File folder3band, File folder8band, File answerFile, File trainingCsv) {
		loadPredictors(rfBuilding, rfBorder, rfPolyMatch);
		if (isLocal) {
			Map<String, List<Building>> buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
			Util.splitImages(buildingsPerImage, new int[] { 60, 30, 10 }, 2);
			testingImages.addAll(buildingsPerImage.keySet());
			Collections.sort(testingImages, new Comparator<String>() {
				public int compare(String a, String b) {
					return Integer.compare(buildingsPerImage.get(b).size(), buildingsPerImage.get(a).size());
				}
			});
		} else {
			File[] files = folder3band.listFiles();
			List<String> ids = new ArrayList<String>();
			String prefix = folder3band.getName() + "_";
			String suffix = ".tif";
			for (File file : files) {
				if (file.getName().startsWith(prefix) && file.getName().endsWith(suffix)) {
					String id = file.getName().substring(prefix.length(), file.getName().length() - suffix.length());
					ids.add(id);
				}
			}
			testingImages.addAll(ids);
		}
		processImages(folderPan, folder3band, folder8band);
		writeAnswer(answerFile, true);
	}

	private void processImages(File folderPan, File folder3band, File folder8band) {
		try {
			System.err.println("Processing Images");
			long t = System.currentTimeMillis();
			final List<String> images = new ArrayList<String>(testingImages);
			final int tot = images.size();
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				threads[i] = new Thread() {
					public void run() {
						while (true) {
							String imageId = null;
							int size = 0;
							synchronized (images) {
								if (images.isEmpty()) break;
								imageId = images.remove(0);
								size = images.size();
							}
							System.err.println("\t" + imageId + " : " + size + "/" + tot);
							processImage(imageId, folderPan, folder3band, folder8band);
						}
					}
				};
				threads[i].start();
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
				threads[i] = null;
			}
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void writeAnswer(File answerFile, boolean checkEmpty) {
		try {
			System.err.println("Writing Answer");
			long t = System.currentTimeMillis();
			BufferedWriter out = new BufferedWriter(new FileWriter(answerFile, true));
			Set<String> seen = new HashSet<String>();
			for (String line : answers) {
				if (line == null) continue;
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

	private void processImage(String imageId, File folderPan, File folder3band, File folder8band) {
		try {
			File imageFilePan = new File(folderPan, folderPan.getName() + "_" + imageId + ".tif");
			File imageFile3band = new File(folder3band, folder3band.getName() + "_" + imageId + ".tif");
			File imageFile8band = new File(folder8band, folder8band.getName() + "_" + imageId + ".tif");
			BufferedImage imgPan = ImageIO.read(imageFilePan);
			BufferedImage img3band = ImageIO.read(imageFile3band);
			BufferedImage img8band = ImageIO.read(imageFile8band);
			MultiChannelImage mci = new MultiChannelImage(imgPan, img3band, img8band);
			double[][][] vals = Util.evalImage(mci, buildingPredictor, borderPredictor);
			double[][] buildingValues = vals[0];
			double[][] borderValues = vals[1];
			PolygonFeatureExtractor pfe = new PolygonFeatureExtractor(buildingValues[0].length, buildingValues.length);
			List<MatchPolygon> candidates = new ArrayList<MatchPolygon>();
			for (int i = 0; i < PolygonFeatureExtractor.borderShifts.length; i++) {
				int borderShift = PolygonFeatureExtractor.borderShifts[i];
				double borderWeight = PolygonFeatureExtractor.borderWeights[i];
				for (double cut : PolygonFeatureExtractor.buildingsCuts) {
					List<Polygon> polygons = Util.findBuildings(buildingValues, borderValues, cut, borderWeight, PolygonFeatureExtractor.buildingsPolyBorder + borderShift,
							PolygonFeatureExtractor.buildingsPolyBorder2 + borderShift);
					for (Polygon polygon : polygons) {
						double prob = polyMatchPredictor.predict(pfe.getFeatures(mci, polygon, buildingValues, borderValues));
						if (prob > minConf / 100.0) {
							candidates.add(new MatchPolygon(prob, polygon));
						}
					}
				}
			}
			double maxInter = 0.1;
			Collections.sort(candidates);
			List<Area> areas = new ArrayList<Area>();
			List<Double> aa = new ArrayList<Double>();
			List<MatchPolygon> added = new ArrayList<MatchPolygon>();
			NEXT: for (int i = 0; i < candidates.size(); i++) {
				Polygon pi = candidates.get(i).polygon;
				double pia = Util.getArea(pi);
				Area ai = new Area(pi);
				for (int j = 0; j < added.size(); j++) {
					Polygon pj = added.get(j).polygon;
					if (pi.getBounds().getMaxX() <= pj.getBounds().getMinX()) continue;
					if (pi.getBounds().getMaxY() <= pj.getBounds().getMinY()) continue;
					if (pj.getBounds().getMaxX() <= pi.getBounds().getMinX()) continue;
					if (pj.getBounds().getMaxY() <= pi.getBounds().getMinY()) continue;
					Area aj = new Area(areas.get(j));
					aj.intersect(ai);
					double aaj = Util.areaVal(aj);
					double pja = aa.get(j);
					if (aaj > maxInter * pia || aaj > maxInter * pja) {
						continue NEXT;
					}
				}
				added.add(candidates.get(i));
				areas.add(ai);
				aa.add(pia);
			}

			for (int i = 0; i < added.size(); i++) {
				Polygon polygon = added.get(i).polygon;
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
				sb.append(added.size() - i);
				synchronized (answers) {
					answers.add(sb.toString());
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void loadPredictors(File rfBuilding, File rfBorder, File rfPolyMatch) {
		try {
			System.err.println("Loading Predictors");
			long t = System.currentTimeMillis();
			buildingPredictor = loadPredictor(rfBuilding);
			borderPredictor = loadPredictor(rfBorder);
			polyMatchPredictor = loadPredictor(rfPolyMatch);
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private RandomForestPredictor loadPredictor(File rfFile) {
		try {
			return RandomForestPredictor.load(rfFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}