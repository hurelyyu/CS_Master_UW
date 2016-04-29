import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.HashMap;

public interface RMIinterface extends Remote{

	  // public HashMap<Object, Object> pairs = new HashMap<Object, Object>();
		// put, get, delete
	void put(String thekey, String thevalue) throws RemoteException;
      //get
	String get(String thekey) throws RemoteException;
      //delete 
    void delete(String thekey) throws RemoteException; 
	
}
