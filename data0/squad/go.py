filenames = ['val.ids.question', 'val.ids.context','val.span', 'val.context','val.answer']
newFilenames = ['val.ids.question_c', 'val.ids.context_c', 'val.span_c', 'val.context_c', 'val.answer_c']
for idx in range(5):
    file1 =  open(filenames[idx], "r")
    file2 = open(newFilenames[idx], "w")
    count = 0
    for i in file1:
        count += 1
        if count <= 100:
            file2.write(i)
        else:
            break
    file1.close()
    file2.close()


