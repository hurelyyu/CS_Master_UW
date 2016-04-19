import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Array;
import java.sql.SQLException;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.*;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;
import javax.swing.SortOrder;
import javax.swing.JComponent;
import javax.swing.JComboBox;


public class moviesGUI extends JFrame implements ActionListener, TableModelListener{
	private JButton btnList, btnSearch, btnAdd, btnSort, btnRate;
	private JPanel panelButtons, panelContent; //内容，目录
	private MOVIES_SQL db;
	private List<movies> list;
	private String[] columnNames = {"Title", "Year", "Director", "Studio", "Category","Rate"};

	private Object[][] data;
	private JTable table;
	private JScrollPane scrollpane;
	private JLabel labTitle;
	private JTextField txtTitle;
       
    
	private JLabel[] txtLabel = new JLabel[5];
	private JTextField[] txtField = new JTextField[5];
 
    private JLabel[] txtLabel2 = new JLabel[5];
	private JTextField[] txtField2 = new JTextField[5];
    
    //search panel，但其实我们没有
    private JButton btnTitleSearch;
	private JPanel panelSearch;
    
 	//sort function
    private JPanel panelSort;
	private JButton btnSortMovieBy; //为Sort添加可选下拉菜单
	private JComboBox selectSort;
    private JLabel labSort;

    private DefaultComboBoxModel model;
    public static String SortListSelection = "ANY";
    public static ArrayList<String> theSortVarieties;

	//add panel
	private JPanel panelAdd;
	private JButton btnAddMovie;

        
    //rate panel
    private JPanel panelRate;
    private JButton btnRateMovie;
	
    public moviesGUI(){
		super("Yaqun_Xiao Movie Webset");

		db = new MOVIES_SQL();
		try {
			list = db.getmovie();

			data = new Object[list.size()][columnNames.length];
			for (int i=0; i<list.size(); i++) {
				data[i][0] = list.get(i).getTitle();
				data[i][1] = list.get(i).getYear();
				data[i][2] = list.get(i).getDirector();
				data[i][3] = list.get(i).getStudio();
				data[i][4] = list.get(i).getCategory();	
                data[i][5] = list.get(i).getRate();	
			}
		} catch (SQLException e) {
			e.printStackTrace();
		}
		createComponts();
		setVisible(true);
		setSize(600,600);

	}
       
    public void sortComboBox(){
        theSortVarieties = new ArrayList<String>();
        JComboBox selectSort = new JComboBox<String>();
        selectSort.addItem("Any");
        for (int i = 0; i < theSortVarieties.size(); i++){
            selectSort.addItem(theSortVarieties.get(i));
            }
        };
        
    private void createComponts(){
		panelButtons = new JPanel(); //JPanel class 
		//List 
		btnList = new JButton("Movie List"); //define button name
		btnList.addActionListener(this);

		//search，但我们其实没有显示这个面板
		btnSearch = new JButton("Movie Search");
		btnSearch.addActionListener(this);

		//add
		btnAdd = new JButton("Add Moive");
		btnAdd.addActionListener(this);

		//sort
		btnSort = new JButton("Sort Movie");
		btnSort.addActionListener(this);

		//Rate
		btnRate = new JButton("Rate Movie");
		btnRate.addActionListener(this);

		//add panel area for these button
		panelButtons.add(btnList);
		panelButtons.add(btnAdd);
		panelButtons.add(btnRate);
		add(panelButtons, BorderLayout.NORTH); //put all buttons on top of our GUI

		//List&Sort Panel
		panelContent = new JPanel();
        
        model = new DefaultComboBoxModel();
        panelContent.setLayout(new BoxLayout(panelContent, BoxLayout.PAGE_AXIS));
        
        btnSortMovieBy = new JButton("Sort");
        btnSortMovieBy.addActionListener(this);
        selectSort = new JComboBox(model);
        labSort = new JLabel("Please Make a Selection: ");
       
        model.addElement("title");
        model.addElement("year");
        model.addElement("director");
        model.addElement("studio");
        model.addElement("category");
        model.addElement("rate");

        table = new JTable(data, columnNames);
		scrollpane = new JScrollPane(table);
        
        selectSort.setSelectedIndex(4);
        selectSort.setMaximumSize(new Dimension(300, 40));
		table.getModel().addTableModelListener(this);

        panelContent.add(scrollpane);        
        panelContent.add(labSort);
        panelContent.add(selectSort);
        panelContent.add(btnSortMovieBy);

        
     
		//Search Panel，但其实我们没用到
		panelSearch = new JPanel();
		labTitle = new JLabel("Enter Title You Want Search");
		txtTitle = new JTextField(40);
		btnTitleSearch = new JButton("Search");
		panelSearch.add(labTitle);
		panelSearch.add(txtTitle);
		panelSearch.add(btnTitleSearch);

		//Add Panel
		panelAdd = new JPanel();
		panelAdd.setLayout(new GridLayout(8,0));
		String labelNames[] = {"Enter Movie Title: ", "Enter Movie Year: ", "Enter Movie Director: ",
	                           "Enter Movie Studio: ", "Enter Movie Category: "};

        for (int i = 0; i < labelNames.length; i++){

        	JPanel panel = new JPanel();
        	txtLabel2[i] = new JLabel(labelNames[i]);
        	txtField2[i] = new JTextField(40);
        	panel.add(txtLabel2[i]);
        	panel.add(txtField2[i]);
        	panelAdd.add(panel);
        }
        btnAddMovie = new JButton("Add to List");
        btnAddMovie.addActionListener(this);
        btnAddMovie.setMaximumSize(new Dimension(100,20));
        panelAdd.add(btnAddMovie);

        add(panelContent, BorderLayout.CENTER);


        
        //Rate Panel
        panelRate = new JPanel();
        panelRate.setLayout(new GridLayout(8,0));
    	String RatinglabelNames[] = {"Enter Movie Title: ", "Enter Movie Year: ", "Enter Movie Rate: "};
        for (int i = 0; i < RatinglabelNames.length; i++){
        	JPanel ratePanel = new JPanel();
        	txtLabel[i] = new JLabel(RatinglabelNames[i]);
        	txtField[i] = new JTextField(40);
        	ratePanel.add(txtLabel[i]);
        	ratePanel.add(txtField[i]);
        	panelRate.add(ratePanel);
        }
        btnRateMovie = new JButton("Rate the movie");
        btnRateMovie.addActionListener(this);
        panelRate.add(btnRateMovie);
        add(panelContent, BorderLayout.CENTER);

    }
    
	
    public static void main(String[] args)
    {
    	new moviesGUI();
    }

   
    @Override
    public void actionPerformed(ActionEvent e){
    	if (e.getSource() == btnList){
    		try{
    			list = db.getmovie();
    		} catch (SQLException e1){
    			e1.printStackTrace();
    		}
            
            data = new Object[list.size()][columnNames.length];
    		for (int i = 0; i < list.size(); i++ ) {
                data[i][0] = list.get(i).getTitle();
                data[i][1] = list.get(i).getYear();
                data[i][2] = list.get(i).getDirector();
                data[i][3] = list.get(i).getStudio();
                data[i][4] = list.get(i).getCategory();	
                data[i][5] = list.get(i).getRate();	
    		}
            
    		panelContent.removeAll();
    		table = new JTable(data, columnNames);
    		table.getModel().addTableModelListener(this);
    		scrollpane = new JScrollPane(table);
            panelContent.add(scrollpane);
    		panelContent.add(labSort);
            panelContent.add(selectSort);
            panelContent.add(btnSortMovieBy);
            panelContent.revalidate();
    		this.repaint();
        
        } else if (e.getSource() == btnSortMovieBy){
            String orderby = (String) selectSort.getSelectedItem();  
            try {
                list = db.sortMovie(orderby);
            } catch (SQLException ex) {
                Logger.getLogger(moviesGUI.class.getName()).log(Level.SEVERE, null, ex);
            }

            data = new Object[list.size()][columnNames.length];
            for (int i = 0; i < list.size(); i++ ) {
                data[i][0] = list.get(i).getTitle();
                data[i][1] = list.get(i).getYear();
                data[i][2] = list.get(i).getDirector();
                data[i][3] = list.get(i).getStudio();
                data[i][4] = list.get(i).getCategory();		
                data[i][5] = list.get(i).getRate();	
            }

    		panelContent.removeAll();
    		table = new JTable(data, columnNames);
    		table.getModel().addTableModelListener(this);
    		scrollpane = new JScrollPane(table);
    		panelContent.add(scrollpane);
    		panelContent.add(labSort);
            panelContent.add(selectSort);
            panelContent.add(btnSortMovieBy);
            panelContent.revalidate();
    		this.repaint();
        } else if (e.getSource() == btnAdd) {
			panelContent.removeAll();
			panelContent.add(panelAdd);
			panelContent.revalidate();
			this.repaint();
        } else if (e.getSource() == btnAddMovie){
                String title = txtField2[0].getText();
                Integer year = Integer.parseInt(txtField2[1].getText());
                Integer rate = 0;
                String director = txtField2[2].getText();
                String studio = txtField2[3].getText();
                String category = txtField2[4].getText();
                db.addmovies(title,year,director,studio,category);
                JOptionPane.showMessageDialog(null, "Added into List Successfully!");
                for (int i=0; i < txtField2.length; i++){
                    txtField2[i].setText("");
                }
        } else if (e.getSource() == btnRate) {
			panelContent.removeAll();
			panelContent.add(panelRate);
			panelContent.revalidate();
			this.repaint();
            
        } else if (e.getSource() == btnRateMovie){
                String title = txtField[0].getText();
                Integer year = Integer.parseInt(txtField[1].getText());
                Integer rate = Integer.parseInt(txtField[2].getText());
                try {
                    db.rateMovie(title, year, rate);
                    JOptionPane.showMessageDialog(null, "Rated Successfully!");
                } catch (SQLException ex) {
                    Logger.getLogger(moviesGUI.class.getName()).log(Level.SEVERE, null, ex);
                }
                
                for (int i=0; i < txtField.length; i++){
                    txtField[i].setText("");
                }
        }
            
    
            
		 
    }

    @Override
    public void tableChanged(TableModelEvent e) {
		int row = e.getFirstRow();
        int column = e.getColumn();
        TableModel model = (TableModel)e.getSource();
        String columnName = model.getColumnName(column);
        Object data = model.getValueAt(row, column);
        
        db.updatemovies(row, columnName, data);
	}
/*
    @Override
    public void actionPerformed(ActionEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }*/


 

}



	







	


