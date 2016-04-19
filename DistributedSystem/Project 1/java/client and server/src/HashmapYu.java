import java.util.HashMap;
import java.util.Map;

public interface HashmapYu{

	//store key value
	public static HashMap<Object, Object> sets = new HashMap<Object,Object>();
	//put(key,value) method in Hashmap
	public void put(Object key, Object value);
	//get(Object key) method in hashmap
	public Object get(Object key) throws ClassCaseException;
	//delete method, which is the remove method in hashmap
	public void delete(Object key) throws ClassCaseException;
	
}