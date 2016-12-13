'''
Created on 2016/10/18

@author: FORBiS01
'''
##########################################################
print('〓〓〓〓数値〓〓〓〓')
# 複素数型などの型も
##########################################################
x=1+2
print(x)
#指数
print(2**3)

#乱数の生成
import random
print( random.random() )

#数学
import math
print( math.pi )

##########################################################
print('〓〓〓〓文字列〓〓〓〓')
#  [']または["]によって記述. バックスラッシュでエスケープ
##########################################################
str1 = 'tes\'t'
print( str1 )

#複数行[''']または["""]
str2 = '''len1
len2'''
print( str2 )

#結合
str3 = str1 + ':' + str1
print( str3 )
print( str1*2 )

#インデントアクセス
print( str1[1] )
#終端からインデントアクセス
print( str1[-1] )

##########################################################
print('〓〓〓〓リスト〓〓〓〓')
##########################################################
#リスト内の任意の値をランダムに取得
import random
List1=[1,2,3,4,5]
print( random.choice(List1) )
# 範囲指定 と ＋結合
MList = List1[1:3]+[11,12]
print( MList )

#ソート
print("リストのソート")
MList.sort(); print( MList )

##########################################################
print('〓〓〓〓タプル〓〓〓〓')
# ( ) で括って記述.必須でない.構造体 (struct) のよう
#定義後の追加がデキない
##########################################################
tuple1 = ('Keisuke', 37)
print( tuple1)
tuple2,tuple3=tuple1
print( tuple2)
print( tuple3)
#インデックスでアクセス
print( tuple1[1] )



