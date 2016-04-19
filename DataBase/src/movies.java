//@Author Yaqun Yu

public class movies
{
	private String title;
	private int year;
	private String director;
	private String studio;
	private String category;
    private int rate;

	public movies(String title, int year, String director, String studio, String category, Integer rate)
	{

		this.title = title;
		this.year = year;
		this.director = director;
		this.studio = studio;
		this.category = category;
        this.rate = rate;
	}

	@Override
	public String toString()
	{ //return the string itself

		return "movie [title=" + title + ", year=" + year + ", director="
			+ director + ", studio=" + studio + ", category=" + category + ", rate="+ rate + "]";

	}

	public String getTitle()
	{
		return title;
	}
	public void setTitle(String title)
	{
		this.title = title;
	}
	public int getYear()
	{
		return year;
	}
	public void setYear(int year)
	{
		this.year = year;
	}
	public String getDirector()
	{
		return director;
	}
	public void setDirector(String director)
	{
		this.director = director;
	}
	public String getStudio()
	{
		return studio;
	}
	public void setStudio(String studio)
	{
		this.studio = studio;
	}
	public String getCategory()
	{
		return category;
	}
	public void setCategory(String category)
	{
		this.category = category;
	}
	
	public Integer getRate()
	{
		return rate;
	}
	public void setRate(Integer category)
	{
		this.rate = rate;
	}
	

}
