import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
train_set = np.array([[-1,[1,2]], [-1,[-1,2]], [1,[-1,-2]]])
coef_matrix = np.empty((3,3))
dot_product_matrix = np.empty((3,3))
y_prod_matrix = np.empty((3,3))


# calculate the dot products of all possible combinations of x's.
for i in range(3):
    for j in range(3):
        dot_product_matrix[i][j] = np.dot(train_set[i][1], train_set[j][1])


# calculate the yi*yj part of the terms coefficient.
for i in range(3):
    for j in range(3):
        y_prod_matrix[i][j] = train_set[i][0] * train_set[j][0]
        
# Apply the Hadamard product for element-wise multiplication so as to combine
# the elements in the matrices to calculate coefficient of the alphas in the 
# objective function.
coef_matrix = y_prod_matrix * dot_product_matrix



# Calculate the coefficient matrix for the derivatives of the cost fuction w.r.t
# each alpha. The coefficient matrix is equal to one.

gauss_matrix = np.full((3,3), 0)
alpha = [0,0,0]
derivative = -1
for x in range(3):
    derivative += 1
    for i in range(3):
        for j in range(3):
            if i == derivative:
                if i == j:
                    gauss_matrix[derivative][j] += 2*coef_matrix[i][j]
                    alpha[j] += 2*coef_matrix[i][j]
                  #  print(f'({i},{j}) {2*coef_matrix[i][j]}', end = "\t")
                else:
                    gauss_matrix[derivative][j] += coef_matrix[i][j]
                    alpha[j] += coef_matrix[i][j]
                   # print(f'({i},{j}) {coef_matrix[i][j]}', end = "\t")
            elif j == derivative:
                #print()
                gauss_matrix[derivative][i] += coef_matrix[i][j]
                alpha[i] += coef_matrix[i][j]
                #print(f'Saving ({i},{j}) {coef_matrix[i][j]}')
    #print()
    #print()
    #print('Next')
    
gauss_matrix = gauss_matrix * -0.5

# Augment matrix to include solultions
gauss_matrix = np.c_[gauss_matrix, [[-1]] * 3]
print(gauss_matrix)


# perform guass-Jordan elim// thus convert to reduced row-echelon form

#connvert to row echelon form
for i in range(3):
    for j in range(3): #i think or (i,3)
        if gauss_matrix[i][i] == 0:
            break
        if i == j:
            gauss_matrix[j] = gauss_matrix[i]/gauss_matrix[i][i]
        else:
            gauss_matrix[j] -= gauss_matrix[i]*gauss_matrix[j][i]
       
    
print(gauss_matrix)
print()
print()
# convert to reduced row echelon form

for i in range(3):
    for j in range(2,1,-1):
        gauss_matrix[i] -= gauss_matrix[i][3-j] * gauss_matrix[3-j]
    break

print(np.round(gauss_matrix,4   ))     


# Assuming that the equation will always be row one in matrix, if it isnt just
# use a row op and put it there.



alphas = [0]*3    


# Get all the direct values of alphas from the matrix
for i in range(1,3):
    alphas[i] = round(gauss_matrix[i][3],4) # remove later
 
print()
print(alphas)

# thid matrix will be the matrix that can be solved fully
fin_matrix = np.full((2,3),0.0)



# find vars in first row and put em in matrix
var_count = 0
for i in range(4):
    if gauss_matrix[0][i] != 0:
        fin_matrix[0][var_count] = gauss_matrix[0][i]
        var_count += 1

print()
print()
print('finny\n',fin_matrix)
print()
print()
# calculating last constraint summ(aiyi) = 0

signs = [99]*3
for i,x in enumerate(train_set):
    signs[i] = x[0]

print(signs)
print(alphas)

for i,x in enumerate(alphas):
    if x != 0:
        signs[i] *= x
        
print(signs)
print()
print()

curr_pos = 0
for i in range(3):
    if abs(signs[i]) != abs(1):
        fin_matrix[1][2] += signs[i]*(-1)
    else:
        fin_matrix[1][curr_pos] = signs[i] #(-1) #this fixes a prob you had cause it moves across the equation
        curr_pos += 1

print('fin\n',fin_matrix)
print('new work\n\n')



#Fin matrix now contains the final solvable matrix

for i in range(2):
    for j in range(i,2):
        if i == j:
            fin_matrix[j] = fin_matrix[j]/fin_matrix[i][i]
        else:
            fin_matrix[j] = fin_matrix[j] - fin_matrix[i] * fin_matrix[j][i]


print(fin_matrix)
print()



for i in range(1,2):
    for j in range(i):
        fin_matrix[j] = fin_matrix[j] - fin_matrix[i]

print(fin_matrix)
print(alphas)
curr = 0
for i, x in enumerate(alphas):
    print(i)
    if x == 0:  
        print(fin_matrix[curr][2])
        curr +=1
        alphas[i-1] = fin_matrix[curr][2]

print(alphas)




'''






for i in range(1,2):
    for j in range(2):
        if fin_matrix[i][i] == 0:
            break
        if i == j:
            fin_matrix[j] = fin_matrix[i]/fin_matrix[i][i]
        else:
            fin_matrix[j] -= fin_matrix[i]*fin_matrix[j][i]  
    print(fin_matrix) 






print('\n\nTest')   
print(fin_matrix) 
print()
  for i in range(2):
    if alphas[i] != 0:
        continue
    else:
        alphas[i] = round(fin_matrix[i][2],4)    
    
print()   
print(fin_matrix) 
print()
print(alphas)

'''  
    

    
    
    
    
    
    
    
    