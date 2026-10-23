# from flask import Flask,render_template
# app=Flask(__name__)
# @app.route('/')
# def index1():
#     return render_template("index.html")

# # @app.route('/about')
# # def about():
# #     return render_template("about.html")

# # @app.route('/contact')
# # def contact():
# #     return render_template("contact.html")
# if __name__=="__main__":
#     app.run(debug=True)
# x=10
# while x>=10 and x<=15:
#     print(x)
#     x+=2
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import sqlite3

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route('/python')
def Python():
    return render_template("python.html")

@app.route("/da")
def data():
    return render_template("data.html")
@app.route('/dashboard')
def link():
     conn = get_db()
      # Total students
     count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    # Total unique courses
    #  course_count = conn.execute("SELECT COUNT(DISTINCT course) FROM students").fetchone()[0]
     course_count = conn.execute("SELECT COUNT(DISTINCT course) FROM students").fetchone()[0]
     conn.close()
     return render_template("dashboard.html", count=count,course_count=course_count)
# @app.route('/student_info')
# def add_student():
#     return render_template("add_student.html")

def get_db():
    conn=sqlite3.connect("database.db")
    conn.row_factory=sqlite3.Row
    return conn

@app.route('/create')

def createTable():
    conn=get_db()
    cursor=conn.cursor()
    # cursor.execute("""
    #              create table if not exists students (
    #                id integer primary key autoincrement,name TEXT NOT NULL,course Text NOT NULL
    #                ) 
    
   
    # """
    # )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            designation TEXT,
            gender TEXT,
            dob TEXT,
            email TEXT,
            mobile TEXT,
            whatsapp TEXT,
            residence TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            pin TEXT,
            photo TEXT,
            course TEXT       
        )
    """)
    conn.commit()
    conn.close()
    return "Table Created Successfully"
    # Insert Option
# @app.route('/student_info',methods=["GET","POST"])
# def index():
#         conn=get_db()
#         if request.method=='POST':
#             name=request.form["name"]
#             course=request.form["course"]
#             conn.execute("INSERT INTO students(name,course) VALUES(?,?)",(name,course))
#             conn.commit()
#         students=conn.execute("select * from students").fetchall()
#         conn.close()
#         return render_template("add_student.html",students=students)

@app.route('/student_info', methods=["GET","POST"])
def index():
    conn = get_db()

    if request.method == 'POST':
        name = request.form["name"]
        designation = "Student"
        gender = request.form["gender"]
        dob = request.form["dob"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        whatsapp = request.form["whatsapp"]
        residence = request.form["residence"]
        address = request.form["address"]
        city = request.form["city"]
        state = request.form["state"]
        pin = request.form["pin"]
        course=request.form["course"]

        # Photo upload
        photo = request.files['photo']
        filename = "default.png"

        if photo and photo.filename != "":
            filename = secure_filename(photo.filename)
            photo.save(os.path.join("static/uploads", filename))

        conn.execute("""
            INSERT INTO students 
            (name, designation, gender, dob, email, mobile, whatsapp, residence, address, city, state, pin, photo,course)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
        """, (name, designation, gender, dob, email, mobile, whatsapp, residence, address, city, state, pin, filename,course))

        conn.commit()

    students = conn.execute("SELECT * FROM students").fetchall()
    conn.close()

    return render_template("add_student.html", students=students)


@app.route('/edit/<int:id>', methods=["GET","POST"])
def edit(id):
    conn = get_db()

    if request.method == "POST":
        name = request.form["name"]
        designation = request.form["designation"]
        email = request.form["email"]
        mobile = request.form["mobile"]

        # Handle photo update
        photo = request.files['photo']

        if photo and photo.filename != "":
            filename = secure_filename(photo.filename)
            photo.save(os.path.join("static/uploads", filename))

            # Update with new photo
            conn.execute("""
                UPDATE students 
                SET name=?, designation=?, email=?, mobile=?, photo=? 
                WHERE id=?
            """, (name, designation, email, mobile, filename, id))
        else:
            # Update without changing photo
            conn.execute("""
                UPDATE students 
                SET name=?, designation=?, email=?, mobile=? 
                WHERE id=?
            """, (name, designation, email, mobile, id))

        conn.commit()
        conn.close()

        return redirect("/student_info")

    # GET request
    student = conn.execute(
        "SELECT * FROM students WHERE id=?",
        (id,)
    ).fetchone()

    conn.close()

    return render_template("edit.html", student=student)

@app.route('/delete/<int:id>')
def delete(id):
  conn=get_db()
  conn.execute("DELETE FROM students where id=?",(id,))
  conn.commit()
  conn.close()
  return redirect("/")



# @app.route("/students")
# def students():
#     return render_template("students.html")

# @app.route("/add_student", methods=["GET", "POST"])
# def add_student():
#     if request.method == "POST":
#         name = request.form['name']
#         course = request.form['course']
#         return redirect("/students")
#     return render_template("add_student.html")
if __name__ == "__main__":
    app.run(debug=True)