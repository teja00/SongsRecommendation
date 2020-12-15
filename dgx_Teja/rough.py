# author: Teja and Anil
# Preprocessing the Data
from tinytag import TinyTag
import os
tag_title = []
song_tags = []
count = 0
songs_dic = {}
array = []
for j in os.listdir('../genre wise all songs'):
    arr = os.listdir('../genre wise all songs/'+j)
    for k in arr:
        arr1 = os.listdir('../genre wise all songs/'+j+'/'+k)
        for i in arr1:
            tag = TinyTag.get('../genre wise all songs/'+j+'/'+k+'/'+i)
            result = []
            if(tag.genre != None):
                if(tag.title in songs_dic):
                    continue
                else:
                    try:
                        song_tags.append(tag.genre)
                        tag_title.append('{}\t{}\t{}\n'.format(count,tag.title, tag.albumartist))
                        songs_dic[tag.title] = count
                        result.append(tag.title)
                    except:
                        pass
                count += 1
        array.append(result)
with open('songs.txt', 'w') as filehandle:
    for listitem in tag_title:
        filehandle.write('%s' % listitem)
file1 = open('./tag_hash.txt', 'r')
Lines = file1.readlines()
dic = {}
x = []
for i in Lines:
    x.append(i.split(', '))
le = int(x[len(x)-1][0])
t = []
for i in x:
    t.append(i[1].rstrip("\n"))

mis = []
for j in set(song_tags):
    a = j.lower()
    if a not in t:
        le = le + 1
        mis.append('\n{}, {}'.format(le,a))

with open('./tag_hash.txt', 'a+') as filehandle:
    for listitem in mis:
        filehandle.write('%s' % listitem)
file1 = open('./tag_hash.txt', 'r')
for i in file1:
    x.append(i.split(', '))
for i in x:
    try:
        a = i[1].rstrip("\n")
    except:
        a = i[1]
    dic[a] = i[0]

with open('songtag.txt', 'w') as filehandle:
    for listitem in song_tags:
        x = listitem.lower()
        filehandle.write('%s\n' % dic[x])

with open('playlist.txt','w') as filehandle:
    for i in array:
        for j in i:
            filehandle.write(str(songs_dic[j])+' ')
        filehandle.write('\n')


