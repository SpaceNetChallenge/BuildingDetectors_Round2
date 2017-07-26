import java.util.Arrays;
import java.util.SplittableRandom;

public class ClassificationTree {
	private static double[] log = new double[1 << 27];

	static {
		for (int tot = 1; tot < log.length; tot++) {
			log[tot] = Math.log(tot);
		}
	}

	private ClassificationNode root;
	private final SplittableRandom rnd;

	ClassificationTree(float[][] features, boolean[] classif, int totRows, int idx, int minRowsPerNode, int maxNodes) {
		long t = System.currentTimeMillis();
		ClassificationNode[] nodes = new ClassificationNode[maxNodes + 2];

		rnd = new SplittableRandom(197209091220L + idx);
		int featuresSteps = features.length;
		int[] weight = new int[totRows];
		for (int i = 0; i < totRows; i++) {
			weight[rnd.nextInt(totRows)]++;
		}
		int numSel = 0;
		for (int i = 0; i < totRows; i++) {
			if (weight[i] > 0) numSel++;
		}
		int[] selRows = new int[numSel];
		numSel = 0;
		int classifCount = 0;
		for (int i = 0; i < totRows; i++) {
			if (weight[i] > 0) {
				selRows[numSel++] = i;
				if (classif[i]) classifCount += weight[i];
			}
		}
		root = new ClassificationNode(classifCount, totRows, impurity(classifCount, totRows), 0, numSel - 1);
		int nodeCnt = 0;
		nodes[nodeCnt++] = root;
		int msg = 2;
		float[] splitVals = new float[10];
		for (int i = 0; i < nodeCnt && nodeCnt < maxNodes; i++) {
			ClassificationNode node = nodes[i];
			if (i == msg) {
				System.err.println("\t\t\t" + idx + "\t" + nodeCnt + " nodes\t" + node.numRows + " rows\t" + (System.currentTimeMillis() - t) / 1000 + "s");
				msg *= 2;
			}
			if (node.isPure() || node.numRows < minRowsPerNode * 2) continue;

			double maxSplitGain = 0;
			float bestSplitVal = 0;
			int bestSplitFeature = -1;

			for (int j = 0; j < featuresSteps; j++) {
				int splitFeature = rnd.nextInt(features.length);
				float[] featuresSplitFeature = features[splitFeature];
				for (int k = 0; k < 10; k++) {
					splitVals[k] = featuresSplitFeature[randomNodeRow(selRows, node, rnd)];
				}
				Arrays.sort(splitVals);
				float splitVal1 = splitVals[0];
				float splitVal2 = splitVals[1];
				float splitVal3 = splitVals[2];
				float splitVal4 = splitVals[3];
				float splitVal5 = splitVals[4];
				float splitVal6 = splitVals[5];
				float splitVal7 = splitVals[6];
				float splitVal8 = splitVals[7];
				float splitVal9 = splitVals[8];
				float splitVal10 = splitVals[9];
				int leftTot1 = 0;
				int leftClassif1 = 0;
				int leftTot2 = 0;
				int leftClassif2 = 0;
				int leftTot3 = 0;
				int leftClassif3 = 0;
				int leftTot4 = 0;
				int leftClassif4 = 0;
				int leftTot5 = 0;
				int leftClassif5 = 0;
				int leftTot6 = 0;
				int leftClassif6 = 0;
				int leftTot7 = 0;
				int leftClassif7 = 0;
				int leftTot8 = 0;
				int leftClassif8 = 0;
				int leftTot9 = 0;
				int leftClassif9 = 0;
				int leftTot10 = 0;
				int leftClassif10 = 0;
				for (int r = node.startRow; r <= node.endRow; r++) {
					int row = selRows[r];
					int w = weight[row];
					float rowVal = featuresSplitFeature[row];
					boolean rowClassif = classif[row];
					if (rowVal < splitVal1) {
						if (rowClassif) leftClassif1 += w;
						leftTot1 += w;
					} else if (rowVal < splitVal2) {
						if (rowClassif) leftClassif2 += w;
						leftTot2 += w;
					} else if (rowVal < splitVal3) {
						if (rowClassif) leftClassif3 += w;
						leftTot3 += w;
					} else if (rowVal < splitVal4) {
						if (rowClassif) leftClassif4 += w;
						leftTot4 += w;
					} else if (rowVal < splitVal5) {
						if (rowClassif) leftClassif5 += w;
						leftTot5 += w;
					} else if (rowVal < splitVal6) {
						if (rowClassif) leftClassif6 += w;
						leftTot6 += w;
					} else if (rowVal < splitVal7) {
						if (rowClassif) leftClassif7 += w;
						leftTot7 += w;
					} else if (rowVal < splitVal8) {
						if (rowClassif) leftClassif8 += w;
						leftTot8 += w;
					} else if (rowVal < splitVal9) {
						if (rowClassif) leftClassif9 += w;
						leftTot9 += w;
					} else if (rowVal < splitVal10) {
						if (rowClassif) leftClassif10 += w;
						leftTot10 += w;
					}
				}
				if (leftTot1 >= minRowsPerNode && node.numRows - leftTot1 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif1, leftTot1, node.classif - leftClassif1, node.numRows - leftTot1);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal1;
					}
				}
				leftTot2 += leftTot1;
				leftClassif2 += leftClassif1;
				if (leftTot2 >= minRowsPerNode && node.numRows - leftTot2 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif2, leftTot2, node.classif - leftClassif2, node.numRows - leftTot2);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal2;
					}
				}
				leftTot3 += leftTot2;
				leftClassif3 += leftClassif2;
				if (leftTot3 >= minRowsPerNode && node.numRows - leftTot3 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif3, leftTot3, node.classif - leftClassif3, node.numRows - leftTot3);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal3;
					}
				}
				leftTot4 += leftTot3;
				leftClassif4 += leftClassif3;
				if (leftTot4 >= minRowsPerNode && node.numRows - leftTot4 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif4, leftTot4, node.classif - leftClassif4, node.numRows - leftTot4);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal4;
					}
				}
				leftTot5 += leftTot4;
				leftClassif5 += leftClassif4;
				if (leftTot5 >= minRowsPerNode && node.numRows - leftTot5 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif5, leftTot5, node.classif - leftClassif5, node.numRows - leftTot5);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal5;
					}
				}
				leftTot6 += leftTot5;
				leftClassif6 += leftClassif5;
				if (leftTot6 >= minRowsPerNode && node.numRows - leftTot6 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif6, leftTot6, node.classif - leftClassif6, node.numRows - leftTot6);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal6;
					}
				}
				leftTot7 += leftTot6;
				leftClassif7 += leftClassif6;
				if (leftTot7 >= minRowsPerNode && node.numRows - leftTot7 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif7, leftTot7, node.classif - leftClassif7, node.numRows - leftTot7);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal7;
					}
				}
				leftTot8 += leftTot7;
				leftClassif8 += leftClassif7;
				if (leftTot8 >= minRowsPerNode && node.numRows - leftTot8 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif8, leftTot8, node.classif - leftClassif8, node.numRows - leftTot8);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal8;
					}
				}
				leftTot9 += leftTot8;
				leftClassif9 += leftClassif8;
				if (leftTot9 >= minRowsPerNode && node.numRows - leftTot9 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif9, leftTot9, node.classif - leftClassif9, node.numRows - leftTot9);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal9;
					}
				}
				leftTot10 += leftTot9;
				leftClassif10 += leftClassif9;
				if (leftTot10 >= minRowsPerNode && node.numRows - leftTot10 >= minRowsPerNode) {
					double splitGain = node.impurity - impurity(leftClassif10, leftTot10, node.classif - leftClassif10, node.numRows - leftTot10);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal10;
					}
				}
			}
			if (bestSplitFeature >= 0) {
				int leftTot = 0;
				int rightTot = 0;
				int leftClassif = 0;
				int rightClassif = 0;
				int endLeft = node.endRow;
				float[] featuresSplitFeature = features[bestSplitFeature];
				for (int r = node.startRow; r <= endLeft; r++) {
					int row = selRows[r];
					int w = weight[row];
					if (featuresSplitFeature[row] < bestSplitVal) {
						if (classif[row]) leftClassif += w;
						leftTot += w;
					} else {
						if (classif[row]) rightClassif += w;
						rightTot += w;
						selRows[r--] = selRows[endLeft];
						selRows[endLeft--] = row;
					}
				}
				node.left = new ClassificationNode(leftClassif, leftTot, impurity(leftClassif, leftTot), node.startRow, endLeft);
				node.right = new ClassificationNode(rightClassif, rightTot, impurity(rightClassif, rightTot), endLeft + 1, node.endRow);
				nodes[nodeCnt++] = node.left;
				nodes[nodeCnt++] = node.right;
				node.splitVal = bestSplitVal;
				node.splitFeature = bestSplitFeature;
			}
		}
		System.err.println("\t\t" + idx + "\t" + nodeCnt + " nodes\t" + (System.currentTimeMillis() - t) / 1000 + "s");

	}

	public ClassificationNode getRoot() {
		return root;
	}

	private final double impurity(int cnt, int tot) {
		if (tot <= 1) return 0;
		double lt = log[tot];
		double val = 0;
		if (cnt > 0) val -= cnt * (log[cnt] - lt) / tot;
		cnt = tot - cnt;
		if (cnt > 0) val -= cnt * (log[cnt] - lt) / tot;
		return val;
	}

	private final double impurity(int cnt1, int tot1, int cnt2, int tot2) {
		return (impurity(cnt1, tot1) * tot1 + impurity(cnt2, tot2) * tot2) / (tot1 + tot2);
	}

	private final int randomNodeRow(int[] rows, ClassificationNode node, SplittableRandom rnd) {
		return rows[rnd.nextInt(node.endRow - node.startRow + 1) + node.startRow];
	}
}
