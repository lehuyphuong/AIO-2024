class Ward:
    def __init__(self):
        self.__people = []

        self.__number_of_doctor = 0

        self.__total_teacher_age = 0
        self.__number_of_teacher = 0
        self.__averge_teacher_year_of_birth = 0

    def add_person(self, person):
        self.__people.append(person)

    def show_people(self):
        for person in self.__people:
            print(person)

    def count_doctor(self):
        self.__number_of_doctor = sum(isinstance(person, Doctor)
                                      for person in self.__people)
        print(f"number of doctors: {self.__number_of_doctor}")

    def sort_age(self):
        self.__people.sort(key=self.get_year_of_birth, reverse=True)
        self.show_people()

    def compute_average_teacher_year_of_birth(self):
        for person in self.__people:
            if isinstance(person, Teacher):
                self.__total_teacher_age += 2024 - person._year_of_birth
                self.__number_of_teacher += 1

        self.__averge_teacher_year_of_birth = self.__total_teacher_age / \
            self.__number_of_teacher
        print(f"Average age teacher: {self.__averge_teacher_year_of_birth}")

    @staticmethod
    def get_year_of_birth(people):
        return people._year_of_birth


class Student(Ward):
    def __init__(self, name, year_of_birth, grade):
        super().__init__()
        self._name = name
        self._year_of_birth = year_of_birth
        self._grade = grade

    def get_grade(self):
        return self._grade

    def set_grade(self, grade):
        self._grade = grade

    def describe(self):
        print(f'name = {self._name}')
        print(f'yob = {str(self._year_of_birth)}')
        print(f'grade = {str(self._grade)}')

    def __str__(self):
        return f"Student(Name: {self._name}, \
            yob: {self._year_of_birth}, \
            grade: {self._grade})"


class Teacher(Ward):
    def __init__(self, name, year_of_birth, subject):
        super().__init__()
        self._name = name
        self._year_of_birth = year_of_birth
        self._subject = subject

    def get_subject(self):
        return self._subject

    def set_subject(self, subject):
        self._subject = subject

    def describe(self):
        print(f'name = {self._name}')
        print(f'yob = {str(self._year_of_birth)}')
        print(f'subject = {str(self._subject)}')

    def __str__(self):
        return f"Teacher Name: {self._name}, \
            yob: {self._year_of_birth}, \
            subject: {self._subject} "


class Doctor(Ward):
    def __init__(self, name, year_of_birth, specialist):
        super().__init__()
        self._name = name
        self._year_of_birth = year_of_birth
        self._specialist = specialist

    def get_specialist(self):
        return self._specialist

    def set_specialist(self, specialist):
        self._specialist = specialist

    def describe(self):
        print(f'name = {self._name}')
        print(f'yob = {str(self._year_of_birth)}')
        print(f'specialist = {str(self._specialist)}')

    def __str__(self):
        return f"Doctor(Name: {self._name}, \
            yob: {self._year_of_birth}, \
            specialist: {self._specialist})"

if __name__ == "__main__":
    # Test
    print("Exercise 2a: ---- Create classes and describe method")
    student1 = Student('John', 2000, 11)
    student1.describe()
    print("---------------")
    teacher1 = Teacher('Mark', 1987, 'Math')
    teacher1.describe()
    print("---------------")
    teacher2 = Teacher('Phil', 1987, 'Science')
    teacher2.describe()
    print("---------------")
    doctor1 = Doctor('Kate', 1980, 'Respiratory')
    doctor1.describe()
    print("---------------")
    doctor2 = Doctor('David', 1981, 'Anatomy')
    doctor2.describe()
    print("---------------")

    print("Exercise 2b: ---- List of members")
    ward = Ward()

    ward.add_person(student1)
    ward.add_person(teacher1)
    ward.add_person(teacher2)
    ward.add_person(doctor1)
    ward.add_person(doctor2)

    ward.show_people()

    print("---------------")
    print("Exercise 2c: ---- Count doctor")
    ward.count_doctor()

    print("---------------")
    print("Exercise 2d: ---- Soft age")
    ward.sort_age()

    print("---------------")
    print("Exercise 2e: ---- Compute average teacher age")
    ward.compute_average_teacher_year_of_birth()
