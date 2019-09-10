import java.sql.*;
import java.util.ArrayList;

public class BookDAO {
    public BookDAO(){}

    Connection connection = DatabaseConnection.getConnection();


    private BookDTO extractBookFromResultSet(ResultSet rs) throws SQLException {
        BookDTO book = new BookDTO();

        book.setID(rs.getInt("book_id"));
        book.setName(rs.getString("book_name"));
        book.setAuthor(rs.getString("author"));

        return book;
    }


    public BookDTO getBook(int id){
        try {
            Statement stmt = connection.createStatement();
            ResultSet rs = stmt.executeQuery("select * from novels where book_id=" + id);

            if (rs.next()){
                return extractBookFromResultSet(rs);
            }
        } catch (SQLException ex){
            ex.printStackTrace();
        }

        return null;
    }


    public ArrayList<BookDTO> getAllBooks(){
        try {
            Statement stmt = connection.createStatement();
            ResultSet rs = stmt.executeQuery("select * from novels");

            ArrayList<BookDTO> books = new ArrayList<>();

            while (rs.next()){
                BookDTO book = extractBookFromResultSet(rs);
                books.add(book);
            }
            return books;
        } catch (SQLException ex){
            ex.printStackTrace();
        }

        return null;
    }


    public boolean insertBook(BookDTO book){
        try {
            PreparedStatement ps  = connection.prepareStatement("insert into novels values (null, ?, ?)");
            ps.setString(1, book.getName());
            ps.setString(2, book.getAuthor());
            int i = ps.executeUpdate();

            if (i == 1){
                return true;
            }

        } catch (SQLException ex){
            ex.printStackTrace();
        }

        return false;
    }


    public boolean updateBook(BookDTO book){
        try {
            PreparedStatement ps  = connection.prepareStatement("update novels set book_name=?, author=? where book_id=?");
            ps.setString(1, book.getName());
            ps.setString(2, book.getAuthor());
            ps.setInt(3, book.getID());
            int i = ps.executeUpdate();

            if (i == 1){
                return true;
            }
        } catch (SQLException ex){
            ex.printStackTrace();
        }

        return false;
    }


    public boolean deleteBook(int id){
        try {
            Statement stmt = connection.createStatement();
            int i = stmt.executeUpdate("delete from novels where book_id=" + id);

            if (i == 1){
                return true;
            }

        } catch (SQLException ex){
            ex.printStackTrace();
        }

        return false;
    }
}
