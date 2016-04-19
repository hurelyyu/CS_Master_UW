/**
 * Created by binrusas on 4/8/14.
 */

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.DatagramPacket;
import java.net.Inet4Address;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;

public class TCPServer {
    private int port;
    private Logger logger ;
    ServerSocket serverSocket;
    private Map<String,String> kvalues;
    public TCPServer(int port) throws IOException {
        this.port = port;
        serverSocket = new ServerSocket(port);
        kvalues = new HashMap<String, String>();
        logger = new Logger("TCP-server-log.txt");
        logger.log("Server start running on : " + Inet4Address.getLocalHost());
        System.out.println("Server start running on : " + Inet4Address.getLocalHost());
    }

    private String serveRequest(String request){
        String [] tokens = request.split(",");
        String command = tokens[0];
        String value;
        try{
            if(command.equals("GET")){
                value =  kvalues.get(tokens[1]);
                if(value != null){
                    return "The value of key (" + tokens[1] + ") is (" + value +")";
                }
            }
            else if(command.equals("PUT")){
                kvalues.put(tokens[1], tokens[2]);
                if(kvalues.containsKey(tokens[1])){
                    return "Server has succeeded to handle request = (" + request + ")";
                }
                else{
                    return "Server has failed to handle request =  (" + request + ")";
                }
            }
            else if(command.equals("DELETE")){
                kvalues.remove(tokens[1]);
                if(!kvalues.containsKey(tokens[1])){
                    return "Server has succeeded to handle request =  (" + request + ")";
                }
                else{
                    return "Server has failed to handle request = (" + request + ")";
                }
            }
        }catch (Exception ex){

        }
        return "Server has failed to handle request = (" + request + ")";
    }

    public void listenAndServeRequests() throws IOException {
        while(true){
            final Socket connectionSocket = serverSocket.accept();
            new Thread(){
                public void run(){
                    try {
                        handleRequests(connectionSocket);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }.start();
        }
    }

    private void handleRequests(Socket connectionSocket) throws IOException {
        String request ;
        BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
        DataOutputStream outToClient = new DataOutputStream(connectionSocket.getOutputStream());
        while(true){
            request = inFromClient.readLine();
            if(request != null){
                logger.log("Server has received request from client: " + connectionSocket.getInetAddress() + "\n" +
                        "The Request is : "+ request);
                System.out.println("Server has received request from client: " + connectionSocket.getInetAddress() + "\n" +
                        "The Request is : "+ request);

                String result = serveRequest(request);
                outToClient.writeBytes(result + '\n');

                logger.log("Server has replied to client IP Address : " + connectionSocket.getInetAddress() +
                        ", port: " + connectionSocket.getPort() + "\nThe reply is : (" + result + ")");
                System.out.println("Server has replied to client IP Address : " + connectionSocket.getInetAddress() +
                        ", port: " + connectionSocket.getPort() + "\nThe reply is : (" + result + ")");
            }
        }
    }

    public void closeLogger(){
        logger.close();
    }

    public String getRidOfAnnoyingChar(DatagramPacket packet){
        String result = new String(packet.getData());
        char[] annoyingchar = new char[1];
        char[] charresult = result.toCharArray();
        result = "";
        for(int i=0;i<charresult.length;i++){
            if(charresult[i]==annoyingchar[0]){
                break;
            }
            result+=charresult[i];
        }
        return result;
    }

    public static void main(String args[]){
        if(args.length < 1){
            System.out.println("Error: port must be provided");
            System.exit(2);
        }
        TCPServer server = null ;
        try{
            int port = Integer.parseInt(args[0]);
            server = new TCPServer(port);
            while(true){
                server.listenAndServeRequests();
            }
        }catch (Exception ex){
            ex.printStackTrace();
            server.closeLogger();
        }
    }
}
