import java.io.IOException;


public interface KVServiceProtocolInterface {
	
	public static enum Request {
		PUT, GET, DELETE
	}
	
	void processRequest() throws IOException;
	
	void put(String the_key, String the_value) throws IOException;
	
	void get(String the_key) throws IOException;
	
	void delete(String the_key) throws IOException;
}
