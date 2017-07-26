import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.Area;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

public class PolygonMatcherTrainer {
	private Map<String, List<Building>> buildingsPerImage;
	private static final int numThreads = SpacenetMain.numThreads;
	private static final int maxTrees = 120;
	private static final int minRowsPerNode = 8;
	private static final int maxSamples = 16_000_000;
	private int totSamples = 0;
	private float[][] polyMatchFeatures = new float[PolygonFeatureExtractor.numFeatures][maxSamples];
	private boolean[] polyMatchClassif = new boolean[maxSamples];
	private RandomForestPredictor buildingPredictor, borderPredictor;

	public void run(String trainingFolder, String dataSet) {
		File trainingCsv = new File(trainingFolder, "summaryData/AOI_" + dataSet + "_Train_Building_Solutions.csv");
		File folderPan = new File(trainingFolder, "PAN");
		File folder3band = new File(trainingFolder, "RGB-PanSharpen");
		File folder8band = new File(trainingFolder, "MUL");
		File rfModel = new File("model", dataSet);
		File rfBuilding = new File(rfModel, "rfBuilding.dat");
		File rfBorder = new File(rfModel, "rfBorder.dat");
		File rfPolyMatch = new File(rfModel, "rfPolyMatch.dat");
		train(trainingCsv, folderPan, folder3band, folder8band, rfBuilding, rfBorder, rfPolyMatch);
	}

	public void train(File trainingCsv, File folderPan, File folder3band, File folder8band, File rfBuilding, File rfBorder, File rfPolyMatch) {
		buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
		Util.splitImages(buildingsPerImage, new int[] { 60, 40, 0 }, 1);
		loadPredictors(rfBuilding, rfBorder);
		processImages(folderPan, folder3band, folder8band);
		buildRandomForest(rfPolyMatch);
	}

	private void buildRandomForest(File rfPolyMatch) {
		try {
			System.err.println("Building Random Forest");
			long t = System.currentTimeMillis();
			RandomForestBuilder.train(polyMatchFeatures, polyMatchClassif, totSamples, maxTrees, rfPolyMatch, numThreads, minRowsPerNode);
			System.err.println("\t RF Poly Match: " + rfPolyMatch.length() + " bytes");
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImages(final File folderPan, final File folder3band, final File folder8band) {
		try {
			System.err.println("Processing Images");
			long t = System.currentTimeMillis();
			final List<String> images = new ArrayList<String>(buildingsPerImage.keySet());
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
							File imageFilePan = new File(folderPan, folderPan.getName() + "_" + imageId + ".tif");
							File imageFile3band = new File(folder3band, folder3band.getName() + "_" + imageId + ".tif");
							File imageFile8band = new File(folder8band, folder8band.getName() + "_" + imageId + ".tif");
							List<Building> buildings = buildingsPerImage.get(imageId);
							System.err.println("\t" + imageId + " : " + size + "/" + tot);
							processImage(imageFilePan, imageFile3band, imageFile8band, buildings);
						}
					}
				};
				threads[i].start();
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
				threads[i] = null;
			}
			System.err.println("\tPoly Match Samples: " + totSamples);
			System.err.println("\t      Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(File imageFilePan, File imageFile3band, File imageFile8band, List<Building> groundTruthBuildings) {
		try {
			BufferedImage imgPan = ImageIO.read(imageFilePan);
			BufferedImage img3band = ImageIO.read(imageFile3band);
			BufferedImage img8band = ImageIO.read(imageFile8band);
			MultiChannelImage mci = new MultiChannelImage(imgPan, img3band, img8band);
			double[][][] vals = Util.evalImage(mci, buildingPredictor, borderPredictor);
			double[][] buildingValues = vals[0];
			double[][] borderValues = vals[1];

			List<Polygon> candidates = new ArrayList<Polygon>();
			for (int i = 0; i < PolygonFeatureExtractor.borderShifts.length; i++) {
				int borderShift = PolygonFeatureExtractor.borderShifts[i];
				double borderWeight = PolygonFeatureExtractor.borderWeights[i];
				for (double cut : PolygonFeatureExtractor.buildingsCuts) {
					candidates.addAll(Util.findBuildings(buildingValues, borderValues, cut, borderWeight, PolygonFeatureExtractor.buildingsPolyBorder + borderShift,
							PolygonFeatureExtractor.buildingsPolyBorder2 + borderShift));
				}
			}

			PolygonFeatureExtractor pfe = new PolygonFeatureExtractor(buildingValues[0].length, buildingValues.length);
			for (Polygon polygon : candidates) {
				float value = (float) iou(polygon, groundTruthBuildings);
				float[] features = pfe.getFeatures(mci, polygon, buildingValues, borderValues);
				synchronized (polyMatchClassif) {
					if (totSamples < maxSamples) {
						polyMatchClassif[totSamples] = value > 0.5;
						for (int i = 0; i < features.length; i++) {
							polyMatchFeatures[i][totSamples] = features[i];
						}
						totSamples++;
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private double iou(Polygon poly, List<Building> buildings) {
		double best = 0;
		Area polyArea = new Area(poly);
		double polyAreaVal = Util.areaVal(polyArea);
		Rectangle polyRect = polyArea.getBounds();
		for (Building building : buildings) {
			Area buildingArea = building.getArea();
			Rectangle buildingRect = buildingArea.getBounds();
			if (!buildingRect.intersects(polyRect)) continue;
			Area interArea = new Area(polyArea);
			interArea.intersect(buildingArea);
			double a = Util.areaVal(interArea);
			double curr = a / (polyAreaVal + building.getAreaVal() - a);
			if (curr > best) best = curr;
		}
		return best;
	}

	private void loadPredictors(File rfBuilding, File rfBorder) {
		try {
			System.err.println("Loading Predictors");
			long t = System.currentTimeMillis();
			buildingPredictor = RandomForestPredictor.load(rfBuilding);
			borderPredictor = RandomForestPredictor.load(rfBorder);
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}