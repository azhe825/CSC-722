from pdb import set_trace
dic={}
p=1
tmp=[]
for x2 in range(2):
    tmp1=[x2]
    case={0:0.12,1:0.24}
    p1=case[x2]
    for x3 in range(3):
        tmp2=tmp1+[x3]
        case={(0,0):0.12,(0,1):0.08,(1,1):0.04,(1,2):0.16}
        try:
            p2=p1*case[tuple(tmp2[-2:])]
        except:
            continue
        for x4 in range(3):
            tmp3=tmp2+[x4]
            case={(0,0):0.06,(0,1):0.04,(1,1):0.02,(1,2):0.32,(2,2):0.12}
            try:
                p3=p2*case[tuple(tmp3[-2:])]
            except:
                continue
            for x5 in range(3):
                tmp4=tmp3+[x5]
                case={(0,0):0.06,(0,1):0.04,(1,1):0.02,(1,2):0.32,(2,2):0.12}
                try:
                    p4=p3*case[tuple(tmp4[-2:])]
                except:
                    continue
                for x6 in range(3):
                    tmp5=tmp4+[x6]
                    case={(0,0):0.06,(0,1):0.04,(1,1):0.02,(1,2):0.24,(2,2):0.09}
                    try:
                        p5=p4*case[tuple(tmp5[-2:])]
                        dic[tuple(tmp5)]=p5
                    except:
                        continue
print max(dic.values())
print max(dic.values())/sum(dic.values())
print dic