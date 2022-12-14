from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    part_1 = False
    part_2 = True

    if part_1:
        #Part 1: Analyse number of epochs and energy
        N_train_samples = train_imgs.shape[0]
        epochs = [10, 15, 20]
        energys = []

        for epoch in epochs:
            rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                             ndim_hidden=500,
                                             is_bottom=True,
                                             image_size=image_size,
                                             is_top=False,
                                             n_labels=10,
                                             batch_size=20, )

            n_iter = int((N_train_samples / rbm.batch_size) * epoch)
            print("Number of iterations:", n_iter)

            #run and train
            rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iter)

            energys.append(np.asarray(rbm.energy_lst))


        plt.plot(np.arange(len(energys[0])), energys[0], label = "epochs:"+str(epochs[0]))
        plt.plot(np.arange(len(energys[1])), energys[1], label = "epochs:"+str(epochs[1]))
        plt.plot(np.arange(len(energys[2])), energys[2], '--', label = "epochs:"+str(epochs[2]))
        plt.ylabel("Energy")
        plt.xlabel("Iterations")
        plt.legend()
        plt.savefig("4_1_Energy.pdf")
        plt.show()

    #Part 2: Analyse average reconstruction loss
    if part_2:
        N_train_samples = train_imgs.shape[0]
        n_units = np.arange(200, 550, 50)
        epoch = 1
        mean_rec_error = []

        #build a new network for different amounts of hidden units
        for n_unit in n_units:
            rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                             ndim_hidden=n_unit,
                                             is_bottom=True,
                                             image_size=image_size,
                                             is_top=False,
                                             n_labels=10,
                                             batch_size=20, )

            n_iter = 1500#int((N_train_samples / rbm.batch_size) * epoch)
            print("Number of iterations:", n_iter)


            rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iter)
            mean_rec_error.append(rbm.mean_rec_loss)

        #plot avg error depending on the number iterations for different n_units
        for error,n_unit in zip(mean_rec_error,n_units):
            plt.plot(error, label = "units:"+str(n_unit))

        plt.ylabel("Average reconstruction loss")
        plt.xlabel("Iterations")
        plt.legend()
        plt.savefig("4_1_mrl.pdf")
        plt.show()


    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
