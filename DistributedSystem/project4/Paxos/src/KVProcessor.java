/**
 * 
 * @author yaqunyu
 *
 */
public interface KVProcessor {

	void kv_put(String key, String val);
	
	void kv_delete(String key);
	
	int kv_size();
}
