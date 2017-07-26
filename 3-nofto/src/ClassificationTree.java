
public class ClassificationTree {
	private static double[] imp = new double[1 << 26];

	static {
		for (int tot = 0; tot < 1 << 13; tot++) {
			for (int cnt = 0; cnt <= tot; cnt++) {
				/*double val = 0;
				if (cnt > 0) {
					double p = cnt / (double) tot;
					val -= p * Math.log(p);
				}
				if (tot - cnt > 0) {
					double p = (tot - cnt) / (double) tot;
					val -= p * Math.log(p);
				}*/
				imp[(tot << 13) | cnt] = Math.sqrt(cnt / (double) tot * (1.0 - cnt / (double) tot));//val;
			}
		}
	}

	private ClassificationNode root;
	private final Random rnd;

	ClassificationTree(float[][] features, boolean[] classif, int idx, int totalTrees) {
		long t = System.currentTimeMillis();
		int SPLIT_STEPS = 7;
		int MIN_ROWS_PER_NODE = 16;
		int MAX_LEVEL = 24;
		int MAX_NODES = 1 << 20;
		ClassificationNode[] nodes = new ClassificationNode[MAX_NODES + 2];

		rnd = new Random(197209091220L + idx);
		int numFeatures = features.length;
		int featuresSteps = (numFeatures + 1) / 2;
		int totRows = classif.length;
		int sampledRows = totRows;//Math.min(100000, totRows);
		int[] weight = new int[totRows];
		for (int i = 0; i < sampledRows/*totRows*/; i++) {
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
		root = new ClassificationNode(classifCount, sampledRows/*totRows*/, impurity(classifCount, sampledRows/*totRows*/), 1, 0, numSel - 1);
		int nodeCnt = 0;
		nodes[nodeCnt++] = root;
		float[] prevSplitVal = new float[SPLIT_STEPS];
		for (int i = 0; i < nodeCnt && nodeCnt < MAX_NODES; i++) {
			ClassificationNode node = nodes[i];
			if (node.isPure() || node.level >= MAX_LEVEL || node.total <= MIN_ROWS_PER_NODE) continue;

			double maxSplitGain = 0;
			float bestSplitVal = 0;
			int bestSplitFeature = -1;

			for (int j = 0; j < featuresSteps; j++) {
				int splitFeature = rnd.nextInt(numFeatures);
				float[] featuresSplitFeature = features[splitFeature];
				NEXT: for (int k = 0; k < SPLIT_STEPS; k++) {
					float splitVal = prevSplitVal[k] = featuresSplitFeature[randomNodeRow(selRows, node, rnd)];
					for (int l = 0; l < k; l++) {
						if (splitVal == prevSplitVal[l]) continue NEXT;
					}
					int leftTot = 0;
					int rightTot = 0;
					int leftClassif = 0;
					int rightClassif = 0;
					for (int r = node.startRow; r <= node.endRow; r++) {
						int row = selRows[r];
						int w = weight[row];
						if (featuresSplitFeature[row] < splitVal) {
							if (classif[row]) leftClassif += w;
							leftTot += w;
						} else {
							if (classif[row]) rightClassif += w;
							rightTot += w;
						}
					}
					if (leftTot < MIN_ROWS_PER_NODE || rightTot < MIN_ROWS_PER_NODE) continue;
					double splitGain = node.impurity - impurity(leftClassif, leftTot, rightClassif, rightTot);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal;
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
				node.left = new ClassificationNode(leftClassif, leftTot, impurity(leftClassif, leftTot), node.level + 1, node.startRow, endLeft);
				node.right = new ClassificationNode(rightClassif, rightTot, impurity(rightClassif, rightTot), node.level + 1, endLeft + 1, node.endRow);
				nodes[nodeCnt++] = node.left;
				nodes[nodeCnt++] = node.right;
				node.splitVal = bestSplitVal;
				node.splitFeature = bestSplitFeature;
			}
		}
		System.err.print("\r\t\t" + (idx+1) + " / " + totalTrees + "\t\t" + nodeCnt + " nodes\t" + (System.currentTimeMillis() - t) / 1000 + "s        ");
	}

	ClassificationTree(float[][] features, float[] values, int idx, int totalTrees) {
		long t = System.currentTimeMillis();
		int SPLIT_STEPS = 9;
		int MIN_ROWS_PER_NODE = 16;
		int MAX_LEVEL = 24;
		int MAX_NODES = 1 << 20;
		ClassificationNode[] nodes = new ClassificationNode[MAX_NODES + 2];
		rnd = new Random(197209091220L + idx);
		int numFeatures = features.length;
		int featuresSteps = (int) (Math.sqrt(numFeatures) * 2 + 1);
		int totRows = values.length;
		int sampledRows = totRows; //Math.min(100000, totRows);
		int[] weight = new int[totRows];
		for (int i = 0; i < sampledRows/*totRows*/; i++) {
			weight[rnd.nextInt(totRows)]++;
		}
		int numSel = 0;
		for (int i = 0; i < totRows; i++) {
			if (weight[i] > 0) numSel++;
		}
		int[] selRows = new int[numSel];
		double sum = 0;
		double sumSquares = 0;
		int tw = 0;
		numSel = 0;
		for (int i = 0; i < totRows; i++) {
			double w = weight[i];
			if (w > 0) {
				selRows[numSel++] = i;
				double v = values[i];
				sum += v * w;
				sumSquares += v * v * w;
				tw += w;
			}
		}
		double average = sum / tw;
		double error = error(tw, sum, sumSquares);

		root = new ClassificationNode(1, 0, numSel - 1, average, error);
		int nodeCnt = 0;
		nodes[nodeCnt++] = root;
		float[] prevSplitVal = new float[SPLIT_STEPS];
		for (int i = 0; i < nodeCnt && nodeCnt < MAX_NODES; i++) {
			ClassificationNode node = nodes[i];
			if (node.impurity == 0 || node.level >= MAX_LEVEL || node.getNumRows() <= MIN_ROWS_PER_NODE) continue;

			double maxSplitGain = 0;
			float bestSplitVal = 0;
			int bestSplitFeature = -1;
			double be1 = 0;
			double be2 = 0;

			for (int j = 0; j < featuresSteps; j++) {
				int splitFeature = rnd.nextInt(numFeatures);
				float[] featuresSplitFeature = features[splitFeature];
				NEXT: for (int k = 0; k < SPLIT_STEPS; k++) {
					float splitVal = prevSplitVal[k] = featuresSplitFeature[randomNodeRow(selRows, node, rnd)];
					for (int l = 0; l < k; l++) {
						if (splitVal == prevSplitVal[l]) continue NEXT;
					}
					double sum1 = 0;
					double sum2 = 0;
					double sumSquares1 = 0;
					double sumSquares2 = 0;
					int w1 = 0;
					int w2 = 0;
					for (int r = node.startRow; r <= node.endRow; r++) {
						int row = selRows[r];
						double v = values[row];
						int w = weight[row];
						double vw = v * w;
						if (featuresSplitFeature[row] < splitVal) {
							sum1 += vw;
							sumSquares1 += v * vw;
							w1 += w;
						} else {
							sum2 += vw;
							sumSquares2 += v * vw;
							w2 += w;
						}
					}
					if (w1 < MIN_ROWS_PER_NODE || w2 < MIN_ROWS_PER_NODE) continue;
					double e1 = error(w1, sum1, sumSquares1);
					double e2 = error(w2, sum2, sumSquares2);
					double splitGain = node.impurity - (e1 + e2);
					if (splitGain > maxSplitGain) {
						maxSplitGain = splitGain;
						bestSplitFeature = splitFeature;
						bestSplitVal = splitVal;
						be1 = e1;
						be2 = e2;
					}
				}
			}
			if (bestSplitFeature >= 0) {
				double sum1 = 0;
				double sum2 = 0;
				int w1 = 0;
				int w2 = 0;
				int endLeft = node.endRow;
				float[] featuresSplitFeature = features[bestSplitFeature];
				for (int r = node.startRow; r <= endLeft; r++) {
					int row = selRows[r];
					double v = values[row];
					int w = weight[row];
					double val = featuresSplitFeature[row];
					if (val < bestSplitVal) {
						sum1 += v * w;
						w1 += w;
					} else {
						sum2 += v * w;
						w2 += w;
						selRows[r--] = selRows[endLeft];
						selRows[endLeft--] = row;
					}
				}
				node.left = new ClassificationNode(node.level + 1, node.startRow, endLeft, sum1 / w1, be1);
				node.right = new ClassificationNode(node.level + 1, endLeft + 1, node.endRow, sum2 / w2, be2);
				nodes[nodeCnt++] = node.left;
				nodes[nodeCnt++] = node.right;
				node.splitVal = (float) Math.min(bestSplitVal, bestSplitVal);
				node.splitFeature = bestSplitFeature;
			}
		}
		System.err.print("\r\t\t" + (idx+1) + " / " + totalTrees + "\t\t" + nodeCnt + " nodes\t" + (System.currentTimeMillis() - t) / 1000 + "s        ");
	}

	public ClassificationNode getRoot() {
		return root;
	}

	private final double error(int n, double sum, double sumSquares) {
		return n == 0 ? 0 : sumSquares - sum * sum / n;
	}

	private final double impurity(int cnt, int tot) {
		if (tot < 8192) return imp[(tot << 13) | cnt];
		/*double val = 0;
		if (cnt > 0) {
			double p = cnt / (double) tot;
			val -= p * Math.log(p);
		}
		if (tot - cnt > 0) {
			double p = (tot - cnt) / (double) tot;
			val -= p * Math.log(p);
		}*/
		return Math.sqrt(cnt / (double) tot * (1.0 - cnt / (double) tot));//val;
	}

	private final double impurity(int cnt1, int tot1, int cnt2, int tot2) {
		return (impurity(cnt1, tot1) * tot1 + impurity(cnt2, tot2) * tot2) / (tot1 + tot2);
	}

	private final int randomNodeRow(int[] rows, ClassificationNode node, Random rnd) {
		return rows[rnd.nextInt(node.endRow - node.startRow + 1) + node.startRow];
	}
}

class Random {
	private static final long mask0 = 0x80000000L;
	private static final long mask1 = 0x7fffffffL;
	private static final long[] mult = new long[] {0,0x9908b0dfL};
	private final long[] mt = new long[624];
	private int idx = 0;

	Random(long seed) {
		init(seed);
	}

	private void init(long seed) {
		mt[0] = seed & 0xffffffffl;
		for (int i = 1; i < 624; i++) {
			mt[i] = 1812433253l * (mt[i - 1] ^ (mt[i - 1] >>> 30)) + i;
			mt[i] &= 0xffffffffl;
		}
	}

	private void generate() {
		for (int i = 0; i < 227; i++) {
			long y = (mt[i] & mask0) | (mt[i + 1] & mask1);
			mt[i] = mt[i + 397] ^ (y >> 1) ^ mult[(int) (y & 1)];
		}
		for (int i = 227; i < 623; i++) {
			long y = (mt[i] & mask0) | (mt[i + 1] & mask1);
			mt[i] = mt[i - 227] ^ (y >> 1) ^ mult[(int) (y & 1)];
		}
		long y = (mt[623] & mask0) | (mt[0] & mask1);
		mt[623] = mt[396] ^ (y >> 1) ^ mult[(int) (y & 1)];
	}

	private long rand() {
		if (idx == 0) generate();
		long y = mt[idx];
		idx = (idx + 1) % 624;
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680l;
		y ^= (y << 15) & 0xefc60000l;
		return y ^ (y >> 18);
	}

	int nextInt(int n) {
		return (int) (rand() % n);
	}
}