// HashMap.java : a basic key,value storage interface.
import java.util.HashMap;

public interface HashMapKV {
	//store the key,value pair
	public static HashMap<Object,Object> pairs = new HashMap<Object,Object>();
	// put, get, delete methods
	public void put(Object key,Object value);
	public Object get(Object key) throws CustomException;
	public void delete(Object key) throws CustomException;
}