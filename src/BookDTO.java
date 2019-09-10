public class BookDTO {
    private int id;
    private String name;
    private String author;

    public BookDTO(){}

    public BookDTO(int id, String name, String author){
        this.id = id;
        this.name = name;
        this.author = author;
    }

    public int getID(){
        return id;
    }

    public void setID(int identity){
        this.id = identity;
    }

    public String getName(){
        return name;
    }

    public void setName(String m_name){
        this.name = m_name;
    }

    public String getAuthor(){
        return author;
    }

    public void setAuthor(String a_name){
        this.author = a_name;
    }
    public String toString(){
        return id+"-"+name+"-"+author;
    }
}
