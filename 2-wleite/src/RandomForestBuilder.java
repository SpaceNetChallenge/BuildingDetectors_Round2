
import java.io.File;

public class RandomForestBuilder {
	public static RandomForestPredictor train(final float[][] features, final boolean[] classif, final int totRows, final int maxTrees, final File out, final int numThreads, final int minRowsPerNode) {
		final int maxNodes = totRows / minRowsPerNode * 2;
		final RandomForestPredictor rf = new RandomForestPredictor(maxTrees, maxNodes, true);
		Thread[] threads = new Thread[numThreads];
		for (int i = 0; i < numThreads; i++) {
			final int idx = i;
			threads[i] = new Thread() {
				public void run() {
					for (int i = idx; i < maxTrees; i += numThreads) {
						ClassificationNode root = new ClassificationTree(features, classif, totRows, i, minRowsPerNode, maxNodes).getRoot();
						synchronized (rf) {
							rf.add(root);
						}
					}
				}
			};
			threads[i].start();
		}
		try {
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
				threads[i] = null;
			}
		} catch (InterruptedException e) {
		}
		try {
			rf.save(out);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return rf;
	}
}