def isprime(num):  # 判断因子是不是素数
    flag = 1
    for i in range(2, num):
        if num % i == 0:
            flag = 0
    return flag  # 如果是素数返回1 不是返0

n = int(input())
for i in range(2, n):
    flag = isprime(i)
    if flag == 1:
        while 1:
            if (n % i == 0) and (i <= n):
                n /= i
                print(i)
            else:
                break
    else:
        pass


