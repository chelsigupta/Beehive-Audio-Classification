import com.mysql.cj.jdbc.Driver;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class DatabaseConnection {
    public static final String URL = "jdbc:mysql://localhost/bookstore";
    public static final String USER = "chelsi";
    public static final String PASS = "Chelsi@123";

    // Connecting to database and returning a connection object
    public static Connection getConnection(){
        try {
            DriverManager.registerDriver(new Driver());
            return DriverManager.getConnection(URL+"?"+

                    "useUnicode=true&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=UTC&"
                    + "user="+USER+"&password="+PASS);
        } catch (SQLException ex){
            throw new RuntimeException("Error: Cannot connect to the database", ex);
        }
    }
}
