
public interface IKVProcessor {

	void kv_put(String key, String val);
	
	void kv_delete(String key);
	
	int kv_size();
}
