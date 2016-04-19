/**
 * Created by Aqeel Bin Rustum on 4/21/14.
 */
import java.rmi.Remote;
import java.rmi.RemoteException;

public interface Server extends Remote {
    boolean put(String key, String value) throws RemoteException;
    String get(String key) throws RemoteException;
    boolean delete(String key) throws RemoteException;

}
