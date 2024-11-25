####29/03/24
########OOP IN PYTHON

class Human:
    def __init__(self,n,o):#constructer
        self.name=n
        self.occupation=o
    def speaks(self):
        print(self.name,"says how are you?")
    def do_work(self):
        if self.occupation=="tennis player":
            print(self.name,"plays tennis.")
        elif self.occupation=="actor":
            print(self.name,"shoots film.")
            
tom=Human("Tom_cruise","actor")
tom.do_work()
tom.speaks()
maria=Human("Maria_sharapova","tennis player")
maria.do_work()
maria.speaks()



############INHERITANCE

class vehicle:
    def gn_use(self):
        print("General use:Transportstion")
class car(vehicle):
    def __init__(self):
        print("I am a car.")
        self.wheels=4
        self.has_roof=True
    def sp_use(self):
        self.gn_use()
        print("Specific use:Vacation,long distance travel by road")
class bike(vehicle):
    def __init__(self):
        print("I am a bike.")
        self.wheels=2
        self.has_roof=False
    def sp_use(self):
        self.gn_use()
        print("Specific use:Racing,short distance travel by road")
                
c=car()
c.sp_use()
b=bike() 
b.sp_use()

###Multiple inheritance
class father():
    def skills(self):
        print("I like programing")
class mother():
    def skills(self):
        print("I like drawing") 
class child(father,mother):
    def skills(self):
        father.skills(self)
        mother.skills(self)
        print("I like drawing") 
    
c=child()
c.skills()       

        
                        