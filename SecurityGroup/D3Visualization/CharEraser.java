
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

public class CharEraser {

 public static void main(String[] args) throws IOException {

  String inputfile = "Nodes.txt";
  String outputfile = "Nodes1.txt";
  int number_of_char_to_erased = 15;

  String line = "";

  File input = new File(inputfile);
  Scanner scan = new Scanner(input);
  File output = new File(outputfile);
  PrintStream print = new PrintStream(output);

  while (scan.hasNext()) {
   line = scan.nextLine();
   //line = line.substring(number_of_char_to_erased);
   //line = line.split("]")[0];
   //for (int i = 1; i == line.split(" ").length-1; i++){
       line = "{\"name\":\"vertex\"" + "," + "\"label\":" + line.split(" ")[1];
       print.println(line);
  // }

  }

  scan.close();
  print.close();
 }
}