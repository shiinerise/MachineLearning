

import random
import string
file = open('2.txt','w')
def generate_random_str(randomlength):
    '''
    string.digits = 0123456789
    string.ascii_letters = 26个小写,26个大写
    '''
    str_list = random.sample(string.digits + string.ascii_letters,randomlength)
    random_str = ''.join(str_list)
    return random_str
for i in range(1000):
    # random_str = ''.join(random.sample(string.digits *5 +string.ascii_letters*4,255))
    random_str= generate_random_str(12)
    file.write(random_str + '\n')
file.close()
