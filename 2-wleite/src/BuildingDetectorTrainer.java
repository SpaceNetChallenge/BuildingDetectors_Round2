import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import javax.imageio.ImageIO;

public class BuildingDetectorTrainer {
	private Map<String, List<Building>> buildingsPerImage;
	private final int numThreads = SpacenetMain.numThreads;
	private static final int minRowsPerNode = 16;
	private final int maxTrees = 60;
	private static final int maxSamples = 100_000_000;
	private int totSamples = 0;
	private float[][] features = new float[BuildingFeatureExtractor.numFeatures][maxSamples];
	private boolean[] borderClassif = new boolean[maxSamples];
	private boolean[] buildingClassif = new boolean[maxSamples];

	public void run(String trainingFolder, String dataSet) {
		File trainingCsv = new File(trainingFolder, "summaryData/AOI_" + dataSet + "_Train_Building_Solutions.csv");
		File folderPan = new File(trainingFolder, "PAN");
		File folder3band = new File(trainingFolder, "RGB-PanSharpen");
		File folder8band = new File(trainingFolder, "MUL");
		File rfModel = new File("model", dataSet);
		if (!rfModel.exists()) rfModel.mkdirs();
		File rfBuilding = new File(rfModel, "rfBuilding.dat");
		File rfBorder = new File(rfModel, "rfBorder.dat");
		train(trainingCsv, folderPan, folder3band, folder8band, rfBuilding, rfBorder);
	}

	public void train(File trainingCsv, File folderPan, File folder3band, File folder8band, File rfBuilding, File rfBorder) {
		buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
		Util.splitImages(buildingsPerImage, new int[] { 60, 40, 0 }, 0);
		processImages(folderPan, folder3band, folder8band);
		buildRandomForests(rfBuilding, rfBorder);
	}

	private void buildRandomForests(File rfBuilding, File rfBorder) {
		try {
			System.err.println("Building Random Forests");
			for (int k = 0; k <= 1; k++) {
				long t = System.currentTimeMillis();
				if (k == 0) {
					RandomForestBuilder.train(features, buildingClassif, totSamples, maxTrees, rfBuilding, numThreads, minRowsPerNode);
					System.err.println("\t   RF Building: " + rfBuilding.length() + " bytes");
					System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
				} else if (k == 1) {
					RandomForestBuilder.train(features, borderClassif, totSamples, maxTrees, rfBorder, numThreads, minRowsPerNode);
					System.err.println("\t     RF Border: " + rfBorder.length() + " bytes");
					System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
				}
			}
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
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				final int start = i;
				threads[i] = new Thread() {
					public void run() {
						for (int j = start; j < images.size(); j += numThreads) {
							String imageId = images.get(j);
							File imageFilePan = new File(folderPan, folderPan.getName() + "_" + imageId + ".tif");
							File imageFile3band = new File(folder3band, folder3band.getName() + "_" + imageId + ".tif");
							File imageFile8band = new File(folder8band, folder8band.getName() + "_" + imageId + ".tif");
							List<Building> buildings = buildingsPerImage.get(imageId);
							System.err.println("\t\t" + (j + 1) + "/" + images.size() + "\t" + imageId);
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
			int nInside = 0;
			int nBorder = 0;
			for (int i = 0; i < totSamples; i++) {
				if (borderClassif[i]) nBorder++;
				if (buildingClassif[i]) nInside++;
			}
			System.err.println("\t          Inside: " + nInside);
			System.err.println("\t          Border: " + nBorder);
			System.err.println("\t         Samples: " + totSamples);
			System.err.println("\t    Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(File imageFilePan, File imageFile3band, File imageFile8band, List<Building> buildings) {
		try {
			BufferedImage img3band = ImageIO.read(imageFile3band);
			int w = img3band.getWidth();
			int h = img3band.getHeight();
			int[][] inside = new int[h][w];
			boolean[][] border = new boolean[h][w];
			int id = 0;
			for (Building building : buildings) {
				id++;
				for (int step = 0; step <= 1; step++) {
					int value = step == 0 ? id : 0;
					List<Polygon> polygons = step == 0 ? building.in : building.out;
					for (Polygon polygon : polygons) {
						Rectangle rc = polygon.getBounds();
						int yMin = Math.max(rc.y, 0);
						int yMax = Math.min(rc.y + rc.height, h);
						int xMin = Math.max(rc.x, 0);
						int xMax = Math.min(rc.x + rc.width, w);
						for (int y = yMin; y < yMax; y++) {
							int[] iy = inside[y];
							for (int x = xMin; x < xMax; x++) {
								if (polygon.contains(x, y)) iy[x] = value;
							}
						}
					}
				}
			}

			int[] queue = new int[w * h * 4];
			int tot = 0;
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int v = inside[y][x];
					boolean b = false;
					if (x > 0 && v != inside[y][x - 1]) b = true;
					else if (x < w - 1 && v != inside[y][x + 1]) b = true;
					else if (y > 0 && v != inside[y - 1][x]) b = true;
					else if (y < h - 1 && v != inside[y + 1][x]) b = true;
					if (b) {
						border[y][x] = true;
						queue[tot++] = y * w + x;
					}
				}
			}

			SplittableRandom rnd = new SplittableRandom(22062012 + imageFile8band.getName().hashCode());
			BufferedImage img8band = ImageIO.read(imageFile8band);
			BufferedImage imgPan = ImageIO.read(imageFilePan);
			MultiChannelImage mci = new MultiChannelImage(imgPan, img3band, img8band);
			int outSubsample = 16;
			int inSubsample = 8;
			int borderSubsample = 1;
			if (id == 0) outSubsample *= 4;
			for (int y = 0; y < h; y++) {
				int[] iy = inside[y];
				boolean[] by = border[y];
				for (int x = 0; x < w; x++) {
					boolean isBorder = by[x];
					boolean isInside = iy[x] > 0;
					int subsample = isBorder ? borderSubsample : isInside ? inSubsample : outSubsample;
					if (rnd.nextInt(subsample) > 0) continue;
					float[] arrFeatures = BuildingFeatureExtractor.getFeatures(mci, x, y, w, h);
					synchronized (borderClassif) {
						if (totSamples < maxSamples) {
							buildingClassif[totSamples] = isInside;
							borderClassif[totSamples] = isBorder;
							for (int i = 0; i < arrFeatures.length; i++) {
								features[i][totSamples] = arrFeatures[i];
							}
							totSamples++;
						}
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}