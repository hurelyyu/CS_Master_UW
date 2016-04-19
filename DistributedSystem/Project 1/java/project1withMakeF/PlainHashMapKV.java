// PlainHashMapKV.java: an implementation of the HashMapKV interface.
import java.util.HashMap;
public class PlainHashMapKV implements HashMapKV {
	// method to put a (key,value) pair in our hashMap pairs
	public void put(Object key, Object value) {
		pairs.put(key,value);
		System.out.println(key+", "+value+" pair is added");
	}

	// method to get the value if the input key is in our hashMap
	public Object get(Object key) throws CustomException {
		if(!pairs.containsKey(key)) throw new CustomException("Error: "+key+" does not exist in KeySet");
		return pairs.get(key);
	}

	// method to delete a key,value pair if the input key is in our hashMap
	public void delete(Object key) throws CustomException {
		// custom defined exception if the key is not in the hashMap
		if(!pairs.containsKey(key)) throw new CustomException("Error: "+key+" does not exist in KeySet");
		else {
			pairs.remove(key);
			System.out.println(key+" and its value are removed from set");
		}
	}
}