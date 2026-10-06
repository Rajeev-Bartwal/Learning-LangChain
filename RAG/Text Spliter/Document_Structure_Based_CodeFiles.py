from langchain_text_splitters import RecursiveCharacterTextSplitter ,  Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
        self.subjects = []

    def add_subject(self, subject):
        self.subjects.append(subject)

    def get_details(self):
        return {
            "name": self.name,
            "age": self.age,
            "grade": self.grade,
            "subjects": self.subjects
        }

    def calculate_average(self, marks):
        if not marks:
            return 0
        return sum(marks) / len(marks)


class Teacher:
    def __init__(self, name, subject, experience):
        self.name = name
        self.subject = subject
        self.experience = experience
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def get_student_count(self):
        return len(self.students)

    def display_info(self):
        print(f"Teacher: {self.name}")
        print(f"Subject: {self.subject}")
        print(f"Experience: {self.experience} years")


def create_student(name, age, grade):
    student = Student(name, age, grade)
    return student


def calculate_total_marks(marks):
    total = 0

    for mark in marks:
        total += mark

    return total


def calculate_percentage(marks):
    if not marks:
        return 0

    total = calculate_total_marks(marks)
    percentage = total / len(marks)

    return percentage


student1 = create_student("Rajeev", 22, "A")
student1.add_subject("Python")
student1.add_subject("Machine Learning")
student1.add_subject("LangChain")

marks = [85, 90, 78, 92, 88]

total_marks = calculate_total_marks(marks)
percentage = calculate_percentage(marks)

print(student1.get_details())
print("Total Marks:", total_marks)
print("Percentage:", percentage)


teacher = Teacher(
    name="John",
    subject="Computer Science",
    experience=8
)

teacher.add_student(student1)
teacher.display_info()

print("Number of Students:", teacher.get_student_count())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(chunk)
    print("-" * 50)