import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties; 


public class MOVIES_SQL {

	private static String username = "root"; //username for mySQL workbench
	private static String password = ""; //did not set any password
	private static String serverName = "127.0.0.1:3306"; //server URL for mySQL workbench
	private static Connection connect; //connection function
	private List<movies> list; //call for movie.java file which contain movie database structure

	public static void createConnection() throws SQLException{

		Properties connectionProps = new Properties();
		connectionProps.put("user", username);
	    connectionProps.put("password", password);

	    connect = DriverManager.getConnection("jdbc:" + "mysql" + "://"
				+ serverName + "/", connectionProps);

	    System.out.println("Successful Connected to database"); //connect with database
	}
    
	public List<movies> getmovie() throws SQLException { 
		if (connect == null) {
			createConnection(); 
		}

        Statement stmt = null;
		String query = "select title, year, director, studio, category, rate "
				+ "from movies_sql.movie "; //select distributes from movie table

		list = new ArrayList<movies>();
		try{
			stmt = connect.createStatement();
			ResultSet resultset = stmt.executeQuery(query);
			while (resultset.next()) { 
			//while there is a next element in resultset

            String title = resultset.getString("title"); //This method returns the char contain in title 
            //distribute until there is no next in title
            int year = resultset.getInt("year");
            String director = resultset.getString("director");
            String studio = resultset.getString("studio");
            String category = resultset.getString("category");
            Integer rate = resultset.getInt("rate");
            movies movie = new movies(title, year, director, studio, category, rate);
            list.add(movie);
            }
		} catch (SQLException e) {
			System.out.println(e);
		} finally {
			if (stmt != null) {
				stmt.close();
			}
		}
		return list;  //put all value into list then return 
    }
    public List<movies> sortMovie(String orderby) throws SQLException{
    
        String sql = "select * from movies_sql.movie order by "+ orderby;
        Statement stmt2 = null;
        
        list = new ArrayList<movies>();
		try{
            stmt2 = connect.createStatement();
            ResultSet resultset2 = stmt2.executeQuery(sql);
			while (resultset2.next()) { 
			//while there is a next element in resultset

				String title = resultset2.getString("title"); //This method returns the char contain in title 
				//distribute until there is no next in title
				int year = resultset2.getInt("year");
				String director = resultset2.getString("director");
				String studio = resultset2.getString("studio");
				String category = resultset2.getString("category");
                Integer rate = resultset2.getInt("rate");
                movies movie = new movies(title, year, director, studio, category, rate);
				list.add(movie);
			}
		} catch (SQLException e) {
			System.out.println(e);
		} finally {
			if (stmt2 != null) {
				stmt2.close();
			}
		}
		return list;  //put all value into list then return 
    };
    
    public void rateMovie(String title, Integer year, Integer rates) throws SQLException {
    	String sqlupdate = "update movies_sql.movie set rate  = ? where title = ? and year = ?";
    	PreparedStatement preparedStatement = null;

    	try {
    		preparedStatement = connect.prepareStatement(sqlupdate);
			preparedStatement.setInt(1, (Integer) rates);
			preparedStatement.setString(2, title);
			preparedStatement.setInt(3, year);
			preparedStatement.executeUpdate();
    		}catch (SQLException e){
    			System.out.println(e);
    			e.printStackTrace();
    		}
        
        
    }

    public List<movies> getmovie(String title){

    	List<movies> filterList = new ArrayList<movies>();
    	try {
    		list = getmovie();
    	} catch (SQLException e) {
    		e.printStackTrace();
    	}
    	for (movies movie : list) {

    		if (movie.getTitle().toLowerCase().contains(title.toLowerCase()))
    		{//cover title and titles in movie to both lowercase and check if they are matching   
    		    filterList.add(movie); //add it to filterlist
    		}
    	}
    	return filterList;
    }

    
    public void addmovies(String title, Integer year, String director, String studio,String category) { //void does not return value, add into list

    	//String addsql = "insert into movies_sql.movie values " + "(" + title + "," + year + "," + director + "," + studio + ","+ category + ","+ 0 +");"; //All parameters in JDBC are represented by the ? symbol, which is known as the parameter marker.
        //must supply values for every parameter before executing the SQL statementS// tring addsql = "insert into movies_sql.movie values " + "(?,?,?,?,?,0);"; //All parameters in JDBC are represented by the ? symbol, which is known as the parameter marker.
    	String addsql = "insert into movies_sql.movie values " + "(?,?,?,?,?,0);"; //All parameters in JDBC are represented by the ? symbol, which is known as the parameter marker.

        PreparedStatement preparedStatement = null; //This statement gives you the flexibility of supplying arguments dynamically.
		
		try {
			preparedStatement = connect.prepareStatement(addsql);
                        preparedStatement.setString(1, title);
			preparedStatement.setInt(2, year);
			preparedStatement.setString(3, director);
			preparedStatement.setString(4, studio);
			preparedStatement.setString(5, category);

			preparedStatement.executeUpdate();
		} catch (SQLException e) {
			System.out.println(e);
			e.printStackTrace();
		} 
    }

    public void updatemovies(int row, String columnName, Object data) { //update list

    	movies movie = list.get(row);
    	String title = movie.getTitle(); //get title from table movie
    	int year = movie.getYear();
    	String sql = "update movies_sql.movie set " + columnName + " = ? where title = ? and year = ?";
    	System.out.println(sql);
    	PreparedStatement preparedStatement = null;

    	try {
    		preparedStatement = connect.prepareStatement(sql);
    		if (data instanceof String) //if data is String type
    			preparedStatement.setString(1,(String)data);
    		else if (data instanceof Integer) // if data type is Integer
				preparedStatement.setInt(1, (Integer) data);
			preparedStatement.setString(2, title);
			preparedStatement.setInt(3, year);
			preparedStatement.executeUpdate();
    		}catch (SQLException e){
    			System.out.println(e);
    			e.printStackTrace();
    		}
    	}

    private void While(boolean next) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}


    
