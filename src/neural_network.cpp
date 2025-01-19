#include "neural_network.h"
using namespace std;

double h = sqrt(std::numeric_limits<double>::epsilon());
uint32_t seed_nn = static_cast<uint32_t>(std::time(0));



///////////////////////////////////////////////////////////////////////////////
/// Functions
///////////////////////////////////////////////////////////////////////////////

// Funcion de activacion sign: n > 0 -> 1, n <= 0 -> -1
double Neural_Network::sign(double n){
    return (n > 0) ? 1.0 : -1.0;
}

// Funcion de activacion relu: Devuelve n si n > 0, 0 en caso contrario.
double Neural_Network::relu(double n){
    return (n > 0) ? n : 0.0;
}

// Funcion de activacion de la tangente hiperbolica: Devuelve (e^n - e^-n)/(e^n + e^-n)
double Neural_Network::tanhiperbolica(double n) {
    return tanh(n);
}

double Neural_Network::squared_loss_function(double h_x,double y){
    return pow((h_x-y),2);
}

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// FORWARD PROPAGATION
///////////////////////////////////////////////////////////////////////////////

// Aplica sobre la matriz señal la funcion de activacion que se le pase como parametro
void Neural_Network::activation(Matrix<double>& output, double (*function_activacion)(double)) const{
    for (auto& row : output) {
        for (auto& val : row) {
            val = function_activacion(val);
        }
    }
}

Matrix<double> Neural_Network::create_matrix(int rows,int cols) const {
    return Matrix<double>(rows, vector<double>(cols));
}

// Genera una matriz de numeros aleatorios entre [-5, 5]
Matrix<double> Neural_Network::genera_matriz_aleatorios(int rows, int columns) const{
    Matrix<double> matriz = create_matrix(rows,columns);
    for (auto& row : matriz) {
        for (auto& val : row) {
            val = (rand() / (double)RAND_MAX) * 10 - 5;
        }
    }
    return matriz;
}

// Calcula la matriz señal
Matrix<double> Neural_Network::multiplicacion_matrices(const Matrix<double>& m1, const Matrix<double>& m2) const{
    int m1_rows = m1.size(), m1_cols = m1[0].size(), m2_cols = m2[0].size();
    Matrix<double> resultado(m1_rows, vector<double>(m2_cols, 0.0));

    for (int i = 0; i < m1_rows; ++i) {
        for (int j = 0; j < m2_cols; ++j) {
            for (int k = 0; k < m1_cols; ++k) {
                resultado[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    return resultado;
}

// Calcula la matriz transpuesta
Matrix<double> Neural_Network::transponer(const Matrix<double>& matriz) const{
    int rows = matriz.size();
    int cols = matriz[0].size();
    Matrix<double> transpuesta(cols, vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transpuesta[j][i] = matriz[i][j];
        }
    }
    return transpuesta;
}

// Añade el bias a la matriz de inputs en la primera fila
void Neural_Network::anyadir_bias(Matrix<double>& inputs) const{
    inputs.insert(inputs.begin(),{1});
}

// Impresión de una matriz, útil para verificar la correcta implementación del algoritmo
void imprimir(const Matrix<double>& matriz) {
    for (const auto& row : matriz) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "----------------------------------" << endl;
}


// Imprime un vector de matrices
void imprimir(const vector<Matrix<double>>& vectores) {
    int index = 0;
    for (const auto& matriz : vectores) {
        cout << "Matriz " << index++ << ":" << endl;
        imprimir(matriz);
    }
}


double Neural_Network::forward(Matrix<double> input) const{
    vector<Matrix<double>> signals;
    vector<Matrix<double>> inputs;
    return forward(input,signals,inputs);
}

// Realiza el forward de la red neuronal
double Neural_Network::forward(Matrix<double> inputs, vector<Matrix<double>>& matrices_signals, vector<Matrix<double>>& matrices_next_inputs) const{
    Matrix<double> senyal;
    int num_capas = neural_network.size();
    int capa_actual = 0;
    anyadir_bias(inputs);
    matrices_next_inputs.push_back(inputs);
    for(Matrix<double> weights : neural_network){ 
        capa_actual++;   
        senyal = multiplicacion_matrices(transponer(weights), inputs);
        matrices_signals.push_back(senyal);
    
//        activation(senyal,sign);
//        activation(senyal,relu);
        activation(senyal,f);  

        inputs = senyal;

        if(capa_actual < num_capas)
            anyadir_bias(inputs);  
        
        matrices_next_inputs.push_back(inputs);
    }
    
    return senyal.at(0).at(0);
} 

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Derivative
///////////////////////////////////////////////////////////////////////////////
double Neural_Network::derivative(func f,double x){
   return (f(x+h)-f(x-h))/(2.0*h);
}

double Neural_Network::derivative_loss(loss_func f,double x,double y){
    return (f(x+h,y)-f(x-h,y))/(2.0*h);
}

///////////////////////////////////////////////////////////////////////////////
/// backpropagation
///////////////////////////////////////////////////////////////////////////////

// Función que se encarga de crear un caso de prueba para probar los algoritmos.
void caso_prueba(vector<Matrix<double>> &neural_network,vector<Matrix<double>> &signal){
   Matrix<double> weights1 = {{0.1,0.2},{0.3,0.4}};
   Matrix<double> weights2 = {{0.2},{1.0},{-3.0}};
   Matrix<double> weights3 = {{1.0},{2.0}};
   neural_network = {weights1,weights2,weights3};
   Matrix<double> signal1 = {{0.7},{1.0}}; 
   Matrix<double> signal2 = {{-1.48}};
   Matrix<double> signal3 = {{-0.8}};
   signal = {signal1,signal2,signal3};
   return;
}

// Función que se encarga de rellenar la matriz que se le pase a partir de la derivada de la función de activación
// respecto a los valores de la matriz señal que se le pase, se asume que la señal posee dimensiones Nx1.
void Neural_Network::fill_sensitive(func f_act,Matrix<double> &sensitive,const Matrix<double> &signal){
   for(const auto &row : signal){
      for(const auto &val : row){
         sensitive.push_back({derivative(f_act,val)});
      }
   }
}

// Función para realizar el producto componente por componente entre 2 vectores.
// Aunque se usa el tipo de dato Matrix se asume que son matrices con dimensiones Nx1.
Matrix<double> Neural_Network::dot_product(Matrix<double> vector1,Matrix<double> vector2){
    if(vector1.size()!=vector2.size()){
        cerr << "Error: Los 2 vectores no poseen las mismas dimensiones." << endl;
    }
    else if(vector1[0].size() > 1 || vector2[0].size() > 1){
        cerr << "Error: Alguno de los vectores no posee la dimensión Nx1." << endl;
    }
    int size = vector1.size();
    Matrix<double> result(size);
    for(int i = 0; i< size;i++){
        result[i].push_back(vector1[i][0]*vector2[i][0]);
    }
    return result;
}

std::vector<Matrix<double>> Neural_Network::backpropagation(const vector<Matrix<double>> &signal){
   std::vector<Matrix<double>> sensitives(neural_network.size());
   //Se obtiene el vector sensitive de la última capa.
   fill_sensitive(f,sensitives[neural_network.size()-1],signal[signal.size()-1]);
   for(int i = neural_network.size()-2;i >= 0;i--){
      Matrix<double> producto = multiplicacion_matrices(neural_network[i+1],sensitives[i+1]);
      // Se debe obtener los elementos desde el 0 hasta d(l) siendo d(l) el número de neuronas de la capa actual
      // sin incluir la neurona extra que se añade para incluir el bias. Es decir, se excluye el bias.
      producto.erase(producto.begin());
      Matrix<double> deriv_signal;
      fill_sensitive(f,deriv_signal,signal[i]);
      sensitives[i] = dot_product(deriv_signal,producto);
   }
   return sensitives;
}

///////////////////////////////////////////////////////////////////////////////
/// Entrenamiento.
///////////////////////////////////////////////////////////////////////////////

void Neural_Network::initialize_gradient(vector<Matrix<double>> &gradient){
    for(auto layer : neural_network){
        Matrix<double> empty_matrix(layer.size(),vector<double>(layer[0].size(),0.0));
        gradient.push_back(empty_matrix);
    }
    return;
}

void Neural_Network::compute_gradient(vector<Matrix<double>> &gradient,double prediction,double exp_output,const vector<Matrix<double>> &sensitives,const vector<Matrix<double>> &inputs){
    for(int i = 0; i < gradient.size();i++){
        Matrix<double> product = multiplicacion_matrices(inputs[i],transponer(sensitives[i]));
        for(int j = 0;j < gradient[i].size();j++){
            for(int k = 0; k < gradient[i][j].size();k++){
                gradient[i][j][k] = derivative_loss(l_func,prediction,exp_output)*product[j][k];
            }
        }
    }
}

void Neural_Network::update_weights(const vector<Matrix<double>> &gradient){
    for(int i = 0;i < neural_network.size();i++){
        for(int j = 0;j < neural_network[i].size();j++){
            for(int k = 0; k < neural_network[i][j].size();k++){
                neural_network[i][j][k] -= learning_rate*gradient[i][j][k];
            }
        }
    }
}

template<typename T,typename U>
void shuffle_data(vector<T> &data_points,vector<U> &outputs){
    for(int i = data_points.size()-1; i >= 0 ;i--){
        int j = rand()%(i+1);
        swap(data_points[i],data_points[j]);
        swap(outputs[i],outputs[j]);
    }
    return;
}


// Función que se encarga de realizar el entrenamiento a partir del conjunto de datos que se le pase.
// Se aplicara el descenso por gradiente estocástico (SGD).
void Neural_Network::SGD_entrenamiento(int epoch,vector<Matrix<double>> &data_points,vector<double> &data_outputs,double convergence_criteria){
    for(int i = 0; i < epoch;i++){
        vector<Matrix<double>> signals;
        vector<Matrix<double>> inputs;
        shuffle_data<Matrix<double>,double>(data_points,data_outputs);
        for(int j = 0; j < data_points.size();j++){
            signals.clear();
            inputs.clear();
            double prediction = forward(data_points[j],signals,inputs);
            if(j >= data_points.size()- 2){
                cout << "PREDICTION: "<< prediction << " EXPECTED OUTPUT: " << data_outputs[j] << endl;
            }
            if(prediction != data_outputs[j]){
                vector<Matrix<double>> gradient;
                initialize_gradient(gradient);
                signals.clear();
                inputs.clear();
                prediction = forward(data_points[j],signals,inputs);
                /*if(l_func(prediction,data_outputs[j])<=convergence_criteria){
                    return;
                }*/
                vector<Matrix<double>> sensitives = backpropagation(signals);
                compute_gradient(gradient,prediction,data_outputs[j],sensitives,inputs);
                update_weights(gradient);
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// DEFINE NETWORK
///////////////////////////////////////////////////////////////////////////////

void create_network(vector<Matrix<double>> &neural_network){
    neural_network.push_back(genera_matriz_aleatorios(129,500));
    neural_network.push_back(genera_matriz_aleatorios(501,500));
    neural_network.push_back(genera_matriz_aleatorios(501,4));
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Leer y Escribir ficheros para la red.
///////////////////////////////////////////////////////////////////////////////

vector<string> get_tokens(string &s, string &delimiters){
    vector<string> tokens;
    size_t last_pos = s.find_first_not_of(delimiters,0);
    size_t pos = s.find_first_of(delimiters,last_pos);
    while(pos != s.npos || last_pos != s.npos){
        tokens.push_back(s.substr(last_pos,pos));
        last_pos = s.find_first_not_of(delimiters,pos);
        pos = s.find_first_of(delimiters,last_pos);
    }
    return tokens;
}

template<class T,class U>
void read_file(vector<Matrix<T>> &data_points,vector<U> &outputs,string &filename){
    ifstream file;
    file.open(filename);
    if(file){
        string s;
        string delimiters = ",\n";
        while(getline(file,s)){
            vector<string> tokens = get_tokens(s,delimiters);
            Matrix<T> data_point;
            for(int i = 0;i < tokens.size()-1;i++){
                data_point.push_back({std::stod(tokens[i])});
            }
            data_points.push_back(data_point);
            outputs.push_back(std::stod(tokens[tokens.size()-1]));
        }
    }
    file.close();
    return;
}


///////////////////////////////////////////////////////////////////////////////