import javax.servlet.annotation.WebServlet;
import java.io.*;
import java.util.ArrayList;

public class StartServlet extends javax.servlet.http.HttpServlet {
    protected void doPost(javax.servlet.http.HttpServletRequest request, javax.servlet.http.HttpServletResponse response) throws javax.servlet.ServletException, IOException {
        String name = request.getParameter("user");
        String pass = request.getParameter("pass");

        if (name.equals("chelsi") && pass.equals("chelsi@123")){
            BookDAO dao = new BookDAO();
            ArrayList<BookDTO> all_books = dao.getAllBooks();
            response.setContentType("text/html");
            PrintWriter out = response.getWriter();
            out.println("<h1>Post</h1>");
            for (BookDTO d: all_books){
                out.println(d.getID()+" "+d.getName()+" "+d.getAuthor());
            }
            out.flush();
        }else {
            response.sendRedirect("welcome.jsp");
        }
    }

    protected void doGet(javax.servlet.http.HttpServletRequest request, javax.servlet.http.HttpServletResponse response) throws javax.servlet.ServletException, IOException {

    }
}

