/**
 * Created by Aqeel Bin Rustum on 4/8/14.
 */

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger {
    PrintWriter printWriter;
    public Logger(String path) throws IOException {
        File file = new File(path);
        if (file.exists()){
            file.delete();
        }
        printWriter = new PrintWriter(new File(path));
    }
    private String getTime(){
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        return "[" + dateFormat.format(date) + "]";
    }
    
    public static String getTimestamp(){
        DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        Date date = new Date();
        return dateFormat.format(date);
    }

    public void log(String message){
        printWriter.println(getTime() + " " + message);
        printWriter.flush();
    }
    public void log(String message, boolean verbose){
        printWriter.println(getTime() + " " + message);
        printWriter.flush();
        if(verbose){
            System.out.println(getTime() + " " + message);
        }
    }

    public void close(){
        printWriter.close();
    }
}