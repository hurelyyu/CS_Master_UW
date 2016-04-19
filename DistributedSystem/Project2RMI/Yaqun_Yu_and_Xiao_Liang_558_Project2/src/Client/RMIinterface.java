
	import java.rmi.Remote;
	import java.rmi.RemoteException;
	import java.util.HashMap;

	public interface RMIinterface extends Remote{

	  // public HashMap<Object, Object> pairs = new HashMap<Object, Object>();
		// put, get, delete
		public int put(Object key, Object value) throws RemoteException;//{
			// TODO Auto-generated method stub
			//return ;
		//}
		public Object get(Object key) throws RemoteException; 
	  	public int delete(Object key) throws RemoteException; 
	}

