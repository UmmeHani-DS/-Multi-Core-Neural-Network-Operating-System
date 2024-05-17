#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cmath>
#include <semaphore.h>
#include <iomanip>

# define Size_of_inputs 4
# define size_of_hidden 7
# define size_of_outputs 4

using namespace std;

pthread_mutex_t Mutex_N;
pthread_mutex_t Mutex_L;
pthread_mutex_t Mutex_LL;
pthread_mutex_t Mutex_Back;
pthread_t completed_threads[3];
pthread_t forward_threads[3];
pthread_t back_threads[3];

struct neuron_struct {
    int pipeFD[2];
    double weights[Size_of_inputs];
    double output;
};

struct layer_struct {
    neuron_struct neurons[size_of_hidden];
    int LayerNumber;
    pthread_t hidden_threads[size_of_hidden];
    sem_t sem_L_forward;
    sem_t sem_L_back;
};

void* computingNeurons(void* args)
{
    neuron_struct* n = (neuron_struct*) args;
    pthread_mutex_lock(&Mutex_N);
    double randomInputs[Size_of_inputs];

    for(int i=0; i<Size_of_inputs; i++)
    {
        randomInputs[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double sumW = 0.0;

    for(int i=0; i<Size_of_inputs; i++)
    {
        sumW += randomInputs[i] * n->weights[i];
    }

    n->output = 1.0 / (1.0 + exp(-sumW));

    pthread_mutex_unlock(&Mutex_N);

    write(n->pipeFD[1], &(n->output), sizeof(n->output)); // write output to pipe 

    pthread_exit(NULL);
}

void* backPropogateLogic(void* args)
{

    pthread_mutex_lock(&Mutex_L);

    neuron_struct* n = (neuron_struct*)args;

    double randomInputs[Size_of_inputs];

    for(int i=0; i<Size_of_inputs; i++)
    {
        randomInputs[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double output_desired = 0.9;

    double learning_rate = 0.01; // just a random learning rate for neural network simulation

    double calculateError = (output_desired - n->output);

    double gradient = calculateError * n->output * (1.0 - n->output);

    //read(n->pipeFD[2], &(n->output), sizeof(n->output));

    for(int i=0; i<Size_of_inputs; i++)
    {
        n->weights[i] += learning_rate * gradient * randomInputs[i];
    }

    pthread_mutex_unlock(&Mutex_L);

    pthread_exit(NULL);
}

void setWeights(neuron_struct* n)
{
    for(int i=0; i<Size_of_inputs; i++)
    {
        n->weights[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void* layer_computation(void* args)
{

    pthread_mutex_lock(&Mutex_LL);

    layer_struct* l = (layer_struct*) args;

    int i=0;
    while(i<size_of_hidden)
    {
        setWeights(&(l->neurons[i]));
        pthread_create(&(l->hidden_threads[i]), NULL, computingNeurons, &(l->neurons[i]));

        if(pipe(l->neurons[i].pipeFD) == -1)
        {
            cout << "Pipe Creation failed..." << endl;
            exit(1);
        }
        i++;
    }

    cout << "Layer Number: " << l->LayerNumber << " Computation: Forward Ouputs: " << endl;

    for(int i=0; i<size_of_hidden; i++)
    {
        pthread_join(l->hidden_threads[i], NULL);
        cout << "Neuron Number: " << i << " Output: " << l->neurons[i].output << endl;
    }

    sem_post(&(l->sem_L_forward));

    pthread_mutex_unlock(&Mutex_LL);

    pthread_exit(NULL);
}

void* backpropogationLayer(void* args)
{

    pthread_mutex_lock(&Mutex_Back);

    layer_struct* l = (layer_struct*)args;

    //sem_wait(&(l->sem_L_forward));

    for(int i=0; i<size_of_hidden; i++)
    {
        pthread_create(&(l->hidden_threads[i]), NULL, backPropogateLogic, &(l->neurons[i]));
    }

    cout << "Layer Number: " << l->LayerNumber << " Computation: Back Propogation Output: " << endl;

    for(int i=0; i<size_of_hidden; i++)
    {
        pthread_join(l->hidden_threads[i], NULL);

        cout << "Neuron Number: " << i << " Output with updated Weights: ";
        for(int c=0; c<Size_of_inputs; c++)
        {
            cout << l->neurons[i].weights[c] << " ";
        }

        cout << endl;
    }

    pthread_mutex_unlock(&Mutex_Back);
    sem_post(&(l->sem_L_back));

    pthread_exit(NULL);
}

void* completed_computations(void* args)
{
    layer_struct* l = (layer_struct*)args;

    cout << "Layer Number: " << l->LayerNumber << " All computations Finished..." << endl;

    pthread_exit(NULL);
}

int main()
{
    pthread_mutex_init(&Mutex_N, NULL);
    pthread_mutex_init(&Mutex_L, NULL);
    pthread_mutex_init(&Mutex_LL, NULL);
    pthread_mutex_init(&Mutex_Back, NULL);

    int status;
    
    layer_struct hiddenL;
    hiddenL.LayerNumber = 2;

    sem_init(&(hiddenL.sem_L_forward), 0, 0);
    sem_init(&(hiddenL.sem_L_back), 0, 0);

    layer_struct inputL;
    inputL.LayerNumber = 1;

    sem_init(&(inputL.sem_L_forward), 0, 0);
    sem_init(&(inputL.sem_L_back), 0, 0);

    layer_struct outputL;
    outputL.LayerNumber = 3;

    sem_init(&(outputL.sem_L_forward), 0, 0);
    sem_init(&(outputL.sem_L_back),0, 0);

    pid_t P1 = fork();

    if(P1 == 0)
    {
        pthread_create(&forward_threads[2], NULL, layer_computation, &inputL);
        pthread_join(forward_threads[2], NULL);
        sem_post(&(inputL.sem_L_forward));
        exit(0);
    }

    else if(P1 > 0)
    {

        waitpid(P1, &status, 0);

        pid_t P2 = fork();

        if(P2 == 0)
        {
            pthread_create(&forward_threads[0], NULL, layer_computation, &hiddenL);
            pthread_join(forward_threads[0], NULL);
            sem_post(&(hiddenL.sem_L_forward));
            exit(0);
        }

        else if(P2 > 0)
        {

            waitpid(P2, &status, 0);

            pthread_create(&forward_threads[1], NULL, layer_computation, &outputL);
            pthread_join(forward_threads[1], NULL);
            sem_post(&(outputL.sem_L_forward));

            pid_t P3 = fork();

            if(P3 == 0)
            {
                pthread_create(&back_threads[1], NULL, backpropogationLayer, &outputL);
                pthread_join(back_threads[1], NULL);
                sem_post(&(outputL.sem_L_back));
                exit(0);
            }
            
            else if(P3 > 0)
            {

                waitpid(P3, &status, 0);

                pid_t P4 = fork();

                if(P4 == 0)
                {
                    pthread_create(&back_threads[0], NULL, backpropogationLayer, &hiddenL);
                    pthread_join(back_threads[0], NULL);
                    sem_post(&(hiddenL.sem_L_back));
                    exit(0);
                }

                else if(P4 > 0)
                {

                    waitpid(P4, &status, 0);

                    pthread_create(&back_threads[2], NULL, backpropogationLayer, &inputL);
                    pthread_join(back_threads[2], NULL);
                    sem_post(&(inputL.sem_L_back));

                    pthread_create(&completed_threads[1], NULL, completed_computations, &inputL);
                    pthread_join(completed_threads[1], NULL);
                    pthread_create(&completed_threads[0], NULL, completed_computations, &hiddenL);
                    pthread_join(completed_threads[0], NULL);
                    pthread_create(&completed_threads[2], NULL, completed_computations, &outputL);
                    pthread_join(completed_threads[2], NULL);

                    exit(0);

                }

            }

        }

    }

    sem_destroy(&(hiddenL.sem_L_forward));
    sem_destroy(&(hiddenL.sem_L_back));
    
    sem_destroy(&(inputL.sem_L_forward));
    sem_destroy(&(inputL.sem_L_back));

    sem_destroy(&(outputL.sem_L_forward));
    sem_destroy(&(outputL.sem_L_back));
    
    pthread_mutex_destroy(&Mutex_Back);
    pthread_mutex_destroy(&Mutex_L);
    pthread_mutex_destroy(&Mutex_LL);
    pthread_mutex_destroy(&Mutex_N);

    pthread_exit(0);

    return 0;
}
