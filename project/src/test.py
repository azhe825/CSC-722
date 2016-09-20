
a=[1,2,3,4]
for x in a:
    print x
    for y in a:
        if y==3:
            a.remove(y)

print a