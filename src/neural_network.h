#include <iostream>
#include <cmath>
#include <vector>
#include <cstdint>
#include <random>
#include <fstream>
#include <algorithm>
#include <ctime>

using namespace std;

#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H

// Definicion de la plantilla Matriz
template <typename T>
using Matrix = vector<vector<T>>;
typedef double(*func)(double);
typedef double(*loss_func)(double,double);
typedef double(*class_loss_func)(Matrix<double>,Matrix<double>);



class Neural_Network{
    public:
        Neural_Network();
        Neural_Network(vector<Matrix<double>> neural_network,double learning_rate);
        Neural_Network(const Neural_Network &copy);
        Neural_Network operator=(const Neural_Network &copy);
        ~Neural_Network();

        static void imprimir(const Matrix<double>& matriz);
        static void imprimir(const vector<Matrix<double>>& vectores);
        void imprimir();

        // Funciones de activación y de coste.
        static double sign(double n);
        static double relu(double n);
        static double tanhiperbolica(double n);
        static double squared_loss_function(double h_x,double y);
        static double cross_entropy_loss(Matrix<double> output, Matrix<double> expected);
        Matrix<double> softmax(Matrix<double> input) const;

        //Funciones para manipular los atributos de la clase.
        void set_funcAct(func f){this->f = f;};
        func get_funcAct(){return f;};
        unsigned int get_num_layers() const {return neural_network.size();};
        void set_layer(int i,Matrix<double> layer){neural_network[i] = layer;};
        Matrix<double> pop_layer();
        void add_layer(Matrix<double> layer);
        void set_learning_rate(double learning_rate){if(learning_rate >0)this->learning_rate = learning_rate;
                                                    else{cerr << "ERROR: Se ha pasado un learning rate invalido." << endl;}};
        double get_learning_rate() const{return learning_rate;}
        
        //Funciones para crear matrices y realizar la multiplicación de matrices.
        static Matrix<double> create_matrix(int rows,int cols);
        static Matrix<double> genera_matriz_aleatorios(int rows, int columns);
        static Matrix<double> multiplicacion_matrices(const Matrix<double>& m1, const Matrix<double>& m2);
        Matrix<double> transponer(const Matrix<double>& matriz) const;

        //Funciones para poder predecir inputs y realizar el entrenamiento.
        void activation(Matrix<double>& output, double (*function_activacion)(double)) const;
        double forward(Matrix<double> input) const; 
        static double derivative(func f,double x);
        void SGD_entrenamiento(int epoch,vector<Matrix<double>> &data_points,vector<unsigned int> &data_outputs);
        double test(vector<Matrix<double>> &data_points,vector<unsigned int> &data_outputs);

        //Funciones para guardar la red en un fichero o construir la red a partir de un fichero.
        void save_network_raw(int num_network=0, string filename="network__raw_");
        void save_network(int num_network=0,string filename="network_");
        void read_network(string filename);
        static void read_file(vector<Matrix<double>> &data_points,vector<unsigned int> &outputs,string &filename);
    private:
        //Atributos.
        vector<Matrix<double>> neural_network;
        class_loss_func l_func;
        func f;
        double learning_rate;

        //Funciones auxiliares.
        void anyadir_bias(Matrix<double>& inputs) const;
        Matrix<double> forward(Matrix<double> inputs, vector<Matrix<double>>& matrices_signals, vector<Matrix<double>>& matrices_next_inputs) const;
        void fill_sensitive(func f_act,Matrix<double> &sensitive,const Matrix<double> &signal);
        Matrix<double> dot_product(Matrix<double> vector1,Matrix<double> vector2);
        std::vector<Matrix<double>> backpropagation(const vector<Matrix<double>> &signal,Matrix<double> last_sensitive);
        void initialize_gradient(vector<Matrix<double>> &gradient);
        void compute_gradient(vector<Matrix<double>> &gradient,const vector<Matrix<double>> &sensitives,const vector<Matrix<double>> &inputs);
        void update_weights(const vector<Matrix<double>> &gradient);
        double get_class(Matrix<double> output) const;
        Matrix<double> get_last_sensitive(Matrix<double> output,unsigned int expected_class) const;
        static double derivative_loss(loss_func f,double x,double y);
};

#endif 