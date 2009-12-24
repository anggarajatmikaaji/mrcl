/**
 * 
 */
package mrcl;

import java.io.DataInput;

import java.io.DataOutput;
import java.io.IOException;


import org.apache.hadoop.io.WritableComparable;

public class MultArgs implements WritableComparable<MultArgs> {
	String _a;
	String _b;
	int _round;

	public MultArgs(String string) {
		String[] words = string.split("/");
		_a = words[0];
		_b = words[1];
		_round = Integer.parseInt(words[2]);
	}

	public MultArgs(String a, String b, int round) {
		_a = a;
		_b = b;
		_round = round;
	}

	public String toString() {
		return String.format("%s/%s/%d", _a, _b, _round);
	}

	public String getA() {
		return _a;
	}

	public String getB() {
		return _b;
	}

	public int getRound() {
		return _round;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		_a = input.readUTF();
		_b = input.readUTF();
		_round = input.readInt();
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeUTF(_a);
		output.writeUTF(_b);
		output.writeInt(_round);
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((_a == null) ? 0 : _a.hashCode());
		result = prime * result + ((_b == null) ? 0 : _b.hashCode());
		result = prime * result + _round;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		MultArgs other = (MultArgs) obj;
		if (_a == null) {
			if (other._a != null)
				return false;
		} else if (!_a.equals(other._a))
			return false;
		if (_b == null) {
			if (other._b != null)
				return false;
		} else if (!_b.equals(other._b))
			return false;
		if (_round != other._round)
			return false;
		return true;
	}

	@Override
	public int compareTo(MultArgs o) {
		return this._round - o._round;
	}
}