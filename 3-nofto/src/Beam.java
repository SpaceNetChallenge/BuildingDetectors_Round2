import java.util.Arrays;

class Beam<T extends Comparable<T>> {
	private int beamWidth;
	private Object[] items;
	private Object[] aux;
	private int size;

	Beam(int beamWidth) {
		this.beamWidth = beamWidth;
		items = new Object[beamWidth];
		aux = new Object[beamWidth];
	}

	void setWidth(int beamWidth) {
		this.beamWidth = beamWidth;
		if (items.length < beamWidth) {
			items = Arrays.copyOf(items, beamWidth);
			aux = new Object[beamWidth];
		}
		if (size > beamWidth) size = beamWidth;
	}

	@SuppressWarnings("unchecked")
	boolean add(T item) {
		if (size >= beamWidth && ((Comparable<T>) items[beamWidth - 1]).compareTo(item) <= 0) return false;
		int pos = Arrays.binarySearch(items, 0, size, item);
		if (pos < 0) pos = -pos - 1;
		else if (items[pos].equals(item)) return false;
		if (pos >= beamWidth) return false;
		if (size < beamWidth) size++;

		System.arraycopy(items, pos, aux, 0, size - pos - 1);
		System.arraycopy(aux, 0, items, pos + 1, size - pos - 1);
		items[pos] = item;
		return true;
	}

	@SuppressWarnings("unchecked")
	T get(int idx) {
		return (T) items[idx];
	}

	@SuppressWarnings("unchecked")
	T last() {
		return (T) items[size - 1];
	}

	int size() {
		return size;
	}

	boolean isFull() {
		return size == beamWidth;
	}

	void clear() {
		Arrays.fill(items, 0, size, null);
		size = 0;
	}

	void remove(int pos) {
		System.arraycopy(items, pos + 1, aux, 0, size - pos - 1);
		System.arraycopy(aux, 0, items, pos, size - pos - 1);
		items[--size] = null;
	}
}