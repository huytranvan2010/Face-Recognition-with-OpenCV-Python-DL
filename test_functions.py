lst = [True, True, False, True]

if True in lst:
    print("anh")

dic = {'anh': 1, 'em': 3, 'oi': 2, 'toi': 10, 'day': 12}

print(max(dic, key=dic.get))

""" Có nhiều cách để sắp xếp dict the value 
Ví dụ sorted(dic.items(), key=lambda x: x[1], reverse=True)"""
new_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

print(new_dic)
print(type(new_dic))
print(new_dic[0][0])

