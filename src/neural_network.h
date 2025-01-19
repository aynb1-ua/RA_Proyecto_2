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

class Neural_Network{
    public:
        Neural_Network();
        Neural_Network(const Neural_Network &copy);
        Neural_Network operator=(const Neural_Network &copy);
        ~Neural_Network();
        static double sign(double n);
        static double relu(double n);
        static double tanhiperbolica(double n);
        static double squared_loss_function(double h_x,double y);
        void activation(Matrix<double>& output, double (*function_activacion)(double)) const;
        Matrix<double> create_matrix(int rows,int cols) const;
        Matrix<double> genera_matriz_aleatorios(int rows, int columns) const;
        Matrix<double> multiplicacion_matrices(const Matrix<double>& m1, const Matrix<double>& m2) const;
        Matrix<double> transponer(const Matrix<double>& matriz) const;
        double forward(Matrix<double> input) const;
        static double derivative(func f,double x);
        static double derivative_loss(loss_func f,double x,double y);
        void SGD_entrenamiento(int epoch,vector<Matrix<double>> &data_points,vector<double> &data_outputs,double convergence_criteria);
    private:
        vector<Matrix<double>> neural_network;
        loss_func l_func;
        func f;
        double learning_rate;
        void anyadir_bias(Matrix<double>& inputs) const;
        double forward(Matrix<double> inputs, vector<Matrix<double>>& matrices_signals, vector<Matrix<double>>& matrices_next_inputs) const;
        void fill_sensitive(func f_act,Matrix<double> &sensitive,const Matrix<double> &signal);
        Matrix<double> dot_product(Matrix<double> vector1,Matrix<double> vector2);
        std::vector<Matrix<double>> backpropagation(const vector<Matrix<double>> &signal);
        void initialize_gradient(vector<Matrix<double>> &gradient);
        void compute_gradient(vector<Matrix<double>> &gradient,double prediction,double exp_output,const vector<Matrix<double>> &sensitives,const vector<Matrix<double>> &inputs);
        void update_weights(const vector<Matrix<double>> &gradient);
};

#endif