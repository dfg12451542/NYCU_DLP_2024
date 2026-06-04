import numpy as np

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def gerenate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x,y,pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] < 0.5:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')

    plt.subplot(1,2,2)
    plt.title('Predict truth', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    
    plt.show()

def ini_w(hidden_layer = 3,n = 100):
    w1 = np.random.randn(hidden_layer,2)
    w2 = np.random.randn(hidden_layer,hidden_layer)
    w3 = np.random.randn(hidden_layer)

    b1 = np.zeros(hidden_layer)
    b2 = np.zeros(hidden_layer)
    b3 = 0

    #x,y = generate_linear(n)
    x,y = gerenate_XOR_easy()

    return w1,w2,w3,b1,b2,b3,x,y

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def d_sigmoid(x):
    return np.multiply(x,1.0-x)

def network():
    hidden_layer = 3
    n = 21
    epsilon = 2
    epochs = 50000

    w1,w2,w3,b1,b2,b3,x,y = ini_w(hidden_layer,n)
    N_points = x.shape[0]
    y2 = y.reshape(N_points)
    #print(x)
    #print(y)
    losses = []

    for iterateion in range(epochs):
        for i in range(1):
    # for iterateion in range(1):
    #     for i in range(1):
            r1 = np.dot(x,w1.T+b1)
            z1 = sigmoid(r1)

            r2 = np.dot(z1,w2.T+b2)
            z2 = sigmoid(r2)

            r3 = np.dot(z2,w3.T+b3)
            z3 = sigmoid(r3)
            Loss = (1/N_points) * np.sum(-y2 * np.log(z3) - (1 - y2) * np.log(1 - z3))
            losses.append(Loss)
            #print('Loss = ',Loss)

            # backpropagation z1 h z2 y

            dLdY = 1/N_points * np.divide(z3 - y2, np.multiply(z3, 1-z3))
            #print(z3.shape)
            #print(dLdY.shape)

            #print(z3.shape)
            #print(y.shape)

            dLdZ3 = np.multiply(dLdY, (z3*(1-z3)))
            dLdW3 = np.dot(z2.T, dLdZ3)

            dLdb3 = np.dot(dLdZ3.T, np.ones(N_points))
            #print(dLdY.shape)
            #print(dLdZ3.shape)
            #print(w3.shape)
            dLdH2 = np.dot(dLdZ3.reshape(N_points, 1), w3.reshape(1, 3))

            dLdZ2 = np.multiply(dLdH2, (z2*(1-z2)))
            #print(w2)
            dLdW2 = np.dot(z1.T, dLdZ2)
            dLdb2 = np.dot(dLdZ2.T, np.ones(N_points))

            dLdH1 = np.dot(dLdZ2.reshape(N_points, 3), w2.reshape(3, 3))

            dLdZ1 = np.multiply(dLdH1, np.multiply(z1, (1-z1)))
            dLdW1 = np.dot(dLdZ1.T, x)
            dLdb1 = np.dot(dLdZ1.T, np.ones(N_points))

            # weights[weight_name] -= epsilon * gradients[weight_name]
            # w1,w2,w3,b1,b2,b3
            b3 -= epsilon * dLdb3
            b2 -= epsilon * dLdb2
            b1 -= epsilon * dLdb1

            #print(w3)
            #print(dLdW3)

            w3 -= epsilon * dLdW3
            w2 -= epsilon * dLdW2
            w1 -= epsilon * dLdW1

    import matplotlib.pyplot as plt
    plt.scatter(range(epochs), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    show_result(x,y,z3)
network()



