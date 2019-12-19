class Test():
    def __init__(self):
        self.a = 123
        self.b = 456

    def forward(self, x):
        x = x * 2
        return x


test = Test()
print(hasattr(test, 'a'))


try:
    print(getattr(test, '_a'))
except:
    print(getattr(test, 'a'))