import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

public class BuildingDetectorTrainer {
	private Map<String, List<Building>> buildingsPerImage;
	private final int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() * 95 / 100);
	private final int maxTrees = 60;
	private List<float[]> features = new ArrayList<float[]>();
	private List<Boolean> borderClassif = new ArrayList<Boolean>();
	private List<Boolean> buildingClassif = new ArrayList<Boolean>();
	private List<Float> distValues = new ArrayList<Float>();
	private Random rnd = new Random(22062012);

	public static void main(String[] args) {
	  String dataDir = "data/AOI_3_Paris_Train";
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
	  File trainingCsv = new File(dataDir + "/summaryData/" + testName + "_Building_Solutions.csv");
		File folder3band = new File(dataDir + "/RGB-PanSharpen");
		File folder8band = new File(dataDir + "/MUL-PanSharpen");
		File rfBuilding = new File(outDir + "rfBuilding.dat");
		File rfBorder = new File(outDir + "rfBorder.dat");
		File rfDist = new File(outDir + "rfDist.dat");
		rfBuilding.getParentFile().mkdirs();
		System.err.println("");
		System.err.println("   Training phase 1 for data '" + testName + "' succesfully started...");
		System.err.println("");
		new BuildingDetectorTrainer().train(trainingCsv, folder3band, folder8band, rfBuilding, rfBorder, rfDist);
	}

	public void train(File trainingCsv, File folder3band, File folder8band, File rfBuilding, File rfBorder, File rfDist) {
		buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
		Util.splitImages(buildingsPerImage, new int[] {65,35,0}, 0);
		processImages(folder3band, folder8band);
		buildRandomForests(rfBuilding, rfBorder, rfDist);
	}

	private void buildRandomForests(File rfBuilding, File rfBorder, File rfDist) {
		try {
			System.err.println("        Building 3 Random Forests, each of " + maxTrees + " trees, with " +  numThreads + " parallel threads...");
			float[][] arrFeatures = new float[features.get(0).length][features.size()];
			for (int i = 0; i < features.size(); i++) {
				float[] a = features.get(i);
				for (int j = 0; j < a.length; j++) {
					arrFeatures[j][i] = a[j];
				}
			}
			features.clear();
			boolean[] classif = null;
			float[] values = null;
			for (int k = 0; k <= 2; k++) {
				long t = System.currentTimeMillis();
				if (k == 0 || k == 1) {
					List<Boolean> lClassif = k == 0 ? buildingClassif : borderClassif;
					classif = new boolean[lClassif.size()];
					for (int i = 0; i < classif.length; i++) {
						classif[i] = lClassif.get(i).booleanValue();
					}
					lClassif.clear();
				} else if (k == 2) {
					List<Float> lValues = distValues;
					values = new float[lValues.size()];
					for (int i = 0; i < classif.length; i++) {
						values[i] = lValues.get(i).floatValue();
					}
					lValues.clear();
				}
				File rfFile = k == 0 ? rfBuilding : k == 1 ? rfBorder : rfDist;
				if (!rfFile.getParentFile().exists()) rfFile.getParentFile().mkdirs();
				if (k == 0 || k == 1) RandomForestBuilder.train(arrFeatures, classif, maxTrees, rfFile, numThreads);
				else RandomForestBuilder.train(arrFeatures, values, maxTrees, rfFile, numThreads);
				if (k == 0) {
	  			System.err.println("");
					System.err.println("\t   RF Building: " + rfBuilding.length() + " bytes        ");
					System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
				} else if (k == 1) {
	  			System.err.println("");
					System.err.println("\t     RF Border: " + rfBorder.length() + " bytes        ");
					System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
				} else if (k == 2) {
	  			System.err.println("");
					System.err.println("\t       RF Dist: " + rfDist.length() + " bytes        ");
					System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
				}
			}
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImages(final File folder3band, final File folder8band) {
		try {
			System.err.println("        Processing Images with " + numThreads + " parallel threads...");
			long t = System.currentTimeMillis();
			final List<String> images = new ArrayList<String>(buildingsPerImage.keySet());
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				final int start = i;
				threads[i] = new Thread() {
					public void run() {
						for (int j = start; j < images.size(); j += numThreads) {
							//if (j!=49) continue;
							String image = images.get(j);
							//File imageFile3band = new File(folder3band, "3band_" + image + ".tif");
							//File imageFile8band = new File(folder8band, "8band_" + image + ".tif");
							File imageFile3band = new File(folder3band, "RGB-PanSharpen_" + image + ".tif");
							File imageFile8band = new File(folder8band, "MUL-PanSharpen_" + image + ".tif");
							List<Building> buildings = buildingsPerImage.get(image);
							processImage(imageFile3band, imageFile8band, buildings);
							/*if (start == 0)*/ System.err.print("\r\t\t" + (j + 1) + "/" + images.size() + "        ");
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
			System.err.println("\t         Samples: " + buildingClassif.size());
			System.err.println("\t    Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(File imageFile3band, File imageFile8band, List<Building> buildings) {
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

			int maxDist = 10;
			int[][] dist = new int[h][w];
			for (int y = 0; y < h; y++) {
				Arrays.fill(dist[y], maxDist + 1);
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
						dist[y][x] = 0;
						queue[tot++] = y * w + x;
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

			BufferedImage img8band = ImageIO.read(imageFile8band);
			MultiChannelImage mci = new MultiChannelImage(img3band, img8band);
			int step = BuildingFeatureExtractor.featTrainingStep;
			if (id == 0) step *= 4;
			for (int y = rnd.nextInt(step); y + BuildingFeatureExtractor.featRectSize <= h; y += step) {
				for (int x = rnd.nextInt(step); x + BuildingFeatureExtractor.featRectSize <= w; x += step) {
					int ic = 0;
					int bc = 0;
					for (int ay = y; ay < y + BuildingFeatureExtractor.featRectSize; ay++) {
						int[] iay = inside[ay];
						boolean[] bay = border[ay];
						for (int ax = x; ax < x + BuildingFeatureExtractor.featRectSize; ax++) {
							if (bay[ax]) bc++;
							else if (iay[ax] > 0) ic++;
						}
					}
					float[] arrFeatures = BuildingFeatureExtractor.getFeatures(mci, x, y, w, h);
					synchronized (borderClassif) {
						buildingClassif.add(ic > 1);
						borderClassif.add(bc > 1);
						distValues.add((float) (dist[y][x] * (ic > 0 ? 1 : -1)));
						features.add(arrFeatures);
					}
				}
			}
			/*
			BufferedImage img2 = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_BGR);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					if (inside[y][x] != 0) img2.setRGB(x, y, 255);
					if (border[y][x]) img2.setRGB(x, y, 256 * 256 * 255);
					int d = dist[y][x];
					if (inside[y][x] == 0) d=-d;
					int c = Color.HSBtoRGB((-d+maxDist)/(float)(3*maxDist+1), 1, 1);
					img2.setRGB(x, y, c);
					img2.setRGB(x + w, y, img3band.getRGB(x, y));
				}
			}
			new ImgViewer(img2);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}