import TextPredict

batch_size=25

with open("../data/seuss.txt") as f:
    for i in range(5):
        inputs = [ord(c) for c in list(f.read(batch_size))]
        print inputs, " "
        one_hot = TextPredict.one_hot(inputs, 256)
        print TextPredict.int_from_one_hot(one_hot)
        if i % 20 == 19:
            print ""