
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

public class test{

 public static void main(String[] args) throws IOException {

  String inputfile = "Links.txt";
  String outputfile = "Links_ego_net.txt";
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
       //line = "{\"name\":\"vertex\"" + "," + "\"label\":" + line.split("\t")[1] + "},";//line for nodes
       line = "{\"source\":" + line.split(" ")[1] + "," + "\"target\":" + line.split(" ")[2] + "," + "\"value\":1" + "},"; // line for links
       print.println(line);
  // }

  }

  scan.close();
  print.close();
 }
}