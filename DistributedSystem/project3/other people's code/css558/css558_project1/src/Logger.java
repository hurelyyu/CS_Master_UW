import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by binrusas on 4/8/14.
 */
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

    public void log(String message){
        printWriter.println(getTime() + " " + message);
        printWriter.flush();
    }

    public void close(){
        printWriter.close();
    }
}
