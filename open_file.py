path = 'data/dream5/chipseq/TF_23_CHIP_51_dinuc.seq'
with open(path, 'r') as f:
    while True:
        content = f.readline()
        # 当读取到文件末尾的时候，跳出循环
        if len(content) == 0:
            break
        print(content, end="")
