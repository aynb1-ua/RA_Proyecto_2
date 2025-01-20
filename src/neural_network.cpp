#include "neural_network.h"
using namespace std;

double h = 1e-5;
uint32_t seed_nn = static_cast<uint32_t>(std::time(0));

///////////////////////////////////////////////////////////////////////////////
/// Create class functions.
///////////////////////////////////////////////////////////////////////////////

Neural_Network::Neural_Network(){
    neural_network = vector<Matrix<double>>();
    f = Neural_Network::tanhiperbolica;
    l_func = Neural_Network::cross_entropy_loss;
    learning_rate = 0.1;
}

Neural_Network::Neural_Network(vector<Matrix<double>> neural_network,double learning_rate){
    this->neural_network = neural_network;
    f = Neural_Network::tanhiperbolica;
    l_func = Neural_Network::cross_entropy_loss;
    this->learning_rate = learning_rate;
}

Neural_Network::Neural_Network(const Neural_Network &copy){
    this->neural_network = copy.neural_network;
    this->f = copy.f;
    this->l_func = copy.l_func;
    this->learning_rate = copy.learning_rate;
}

Neural_Network Neural_Network::operator=(const Neural_Network &copy){
    Neural_Network result(copy);
    return result;
}

Neural_Network::~Neural_Network(){
    f = NULL;
    l_func = NULL;
}

///////////////////////////////////////////////////////////////////////////////

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

double find_max(Matrix<double> input){
    double max = 0.0;
    for(int i = 0;i< input.size();i++){
        if(input[i][0]>max){
            max = input[i][0];
        }
    }
    return max;
}

Matrix<double> Neural_Network::softmax(Matrix<double> input) const{
    double sum = 0.0;
    Matrix<double> result;
    double max = find_max(input);
    for (int i = 0;i < input.size();i++){
        sum += exp(input[i][0]-max);
    }
    for(int i = 0;i < input.size();i++){
        result.push_back({exp(input[i][0]-max)/sum});
    }
    return result;
}

double Neural_Network::cross_entropy_loss(Matrix<double> output, Matrix<double> expected){
    if(output.size() != expected.size()){
        cerr << "Cross entropy loss: output and expected have different size" << endl;
        return -1.0;
    }
    double result = 0.0;
    for(int i = 0;i < output.size();i++){
        result += (-1.0)*(expected[i][0]*log(output[i][0]));
    }
    return result;
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


// Calcula la matriz señal
Matrix<double> Neural_Network::multiplicacion_matrices(const Matrix<double>& m1, const Matrix<double>& m2){
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
void Neural_Network::imprimir(const Matrix<double>& matriz) {
    for (const auto& row : matriz) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "----------------------------------" << endl;
}


// Imprime un vector de matrices
void Neural_Network::imprimir(const vector<Matrix<double>>& vectores) {
    int index = 0;
    for (const auto& matriz : vectores) {
        cout << "Matriz " << index++ << ":" << endl;
        imprimir(matriz);
    }
}

void Neural_Network::imprimir(){
    imprimir(neural_network);
}

// Función que devuelve la posición del vector con mayor probabilidad.
double Neural_Network::get_class(Matrix<double> output) const{
    double max = 0.0;
    int pos = 0;
    for(int i = 0;i < output.size();i++){
        if(output[i][0] > max){
            max = output[i][0];
            pos = i;
        }
    }
    return pos;
}

double Neural_Network::forward(Matrix<double> input) const{
    vector<Matrix<double>> signals;
    vector<Matrix<double>> inputs;
    return get_class(forward(input,signals,inputs));
}

// Realiza el forward de la red neuronal
Matrix<double> Neural_Network::forward(Matrix<double> inputs, vector<Matrix<double>>& matrices_signals, vector<Matrix<double>>& matrices_next_inputs) const{
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
        if(capa_actual < num_capas){
            activation(senyal,f);
        } 
        else{
            senyal = softmax(senyal);
            imprimir(senyal);
        } 

        inputs = senyal;

        if(capa_actual < num_capas)
            anyadir_bias(inputs);  
        
        matrices_next_inputs.push_back(inputs);
    }    
    return inputs;
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

std::vector<Matrix<double>> Neural_Network::backpropagation(const vector<Matrix<double>> &signal,Matrix<double> last_sensitive){
   std::vector<Matrix<double>> sensitives(neural_network.size());
   //Se obtiene el vector sensitive de la última capa.
   //fill_sensitive(f,sensitives[neural_network.size()-1],signal[signal.size()-1]);
   sensitives[neural_network.size()-1] = last_sensitive;
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

void Neural_Network::compute_gradient(vector<Matrix<double>> &gradient,const vector<Matrix<double>> &sensitives,const vector<Matrix<double>> &inputs){
    for(int i = 0; i < gradient.size();i++){
        Matrix<double> product = multiplicacion_matrices(inputs[i],transponer(sensitives[i]));
        for(int j = 0;j < gradient[i].size();j++){
            for(int k = 0; k < gradient[i][j].size();k++){
                gradient[i][j][k] = product[j][k];
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

Matrix<double> Neural_Network::get_last_sensitive(Matrix<double> output,unsigned int expected_class) const{
    Matrix<double> last_sensitive;
    for(unsigned int i = 0;i < output.size();i++){
        if(i == expected_class){
            last_sensitive.push_back({output[i][0]-1.0});
        }
        else{
            last_sensitive.push_back(output[i]);
        }
    }
    return last_sensitive;
}


// Función que se encarga de realizar el entrenamiento a partir del conjunto de datos que se le pase.
// Se aplicara el descenso por gradiente estocástico (SGD).
void Neural_Network::SGD_entrenamiento(int epoch,vector<Matrix<double>> &data_points,vector<unsigned int> &data_outputs){
    for(int i = 0; i < epoch;i++){
        double error = 0.0;
        vector<Matrix<double>> signals;
        vector<Matrix<double>> inputs;
        shuffle_data<Matrix<double>,unsigned int>(data_points,data_outputs);
        for(int j = 0; j < data_points.size();j++){
            signals.clear();
            inputs.clear();
            cout << "DATA POINT: " << data_points[j][0][0] << " EXPECTED OUTPUT: " << data_outputs[j] << endl;
            Matrix<double> prediction = forward(data_points[j],signals,inputs);
            if(j >= data_points.size()- 2){
                cout << "PREDICTION: "<< get_class(prediction) << " EXPECTED OUTPUT: " << data_outputs[j] << endl;
            }
            if(get_class(prediction) != data_outputs[j]){
                vector<Matrix<double>> gradient;
                initialize_gradient(gradient);
                signals.clear();
                inputs.clear();
                prediction = forward(data_points[j],signals,inputs);
                Matrix<double> expected_output(prediction.size(),{0});
                expected_output[data_outputs[j]] = {1};
                error += l_func(prediction,expected_output);
                Matrix<double> last_sensitive = get_last_sensitive(prediction,data_outputs[j]);
                cout << "LAST SENSITIVE ITERATION: " << j << endl;
                imprimir(last_sensitive);
                vector<Matrix<double>> sensitives = backpropagation(signals,last_sensitive);
                compute_gradient(gradient,sensitives,inputs);
                update_weights(gradient);
            }
        }
        cout << "Epoca: " << i << " Error: " << error << endl;
    }
}

// Función que se encarga de devolver el error obtenido al evaluar los datos de test.
double Neural_Network::test(vector<Matrix<double>> &data_points,vector<unsigned int> &data_outputs){
    double error = 0.0;
    for(int i = 0;i < data_points.size();i++){
        vector<Matrix<double>> signals;
        vector<Matrix<double>> inputs;
        Matrix<double> prediction = forward(data_points[i],signals,inputs);
        Matrix<double> expected_output(prediction.size(),{0});
        expected_output[data_outputs[i]] = {1};
        error += l_func(prediction,expected_output);
    }
    return error;
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// DEFINE NETWORK
///////////////////////////////////////////////////////////////////////////////

Matrix<double> Neural_Network::create_matrix(int rows,int cols) {
    return Matrix<double>(rows, vector<double>(cols));
}

// Genera una matriz de numeros aleatorios entre [-5, 5]
Matrix<double> Neural_Network::genera_matriz_aleatorios(int rows, int columns){
    Matrix<double> matriz = create_matrix(rows,columns);
    for (auto& row : matriz) {
        for (auto& val : row) {
            val = (rand() / (double)RAND_MAX) * 10 - 5;
        }
    }
    return matriz;
}

void Neural_Network::add_layer(Matrix<double> layer){
    neural_network.push_back(layer);
}

Matrix<double> Neural_Network::pop_layer(){
    Matrix<double> last_layer = neural_network[neural_network.size()-1];
    neural_network.pop_back();
    return last_layer;
}

void create_network(Neural_Network &neural_network){
    neural_network.add_layer(Neural_Network::genera_matriz_aleatorios(2,3));
    neural_network.add_layer(Neural_Network::genera_matriz_aleatorios(4,2));
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

void Neural_Network::read_file(vector<Matrix<double>> &data_points,vector<unsigned int> &outputs,string &filename){
    ifstream file;
    file.open(filename);
    if(file){
        string s;
        string delimiters = ",\n";
        while(getline(file,s)){
            vector<string> tokens = get_tokens(s,delimiters);
            Matrix<double> data_point;
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


void Neural_Network::save_network_raw(int num_network, string filename){
    filename += to_string(num_network) + ".txt";
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "No se pudo abrir el archivo para guardar la red" << endl;
    }
    int capa = 0;
   
    file << "{";
    for (const Matrix<double>& peso : neural_network) {
        file << "{";
        for(size_t rows = 0; rows < peso.size(); rows++){
            file << "{";
            for(size_t columns = 0; columns < peso[rows].size(); columns++){
                if (columns + 1 != peso[rows].size())
                    file << peso[rows][columns] << ",";
                else
                    file << peso[rows][columns];    
            }
            if(rows+1 !=peso.size())
                file << "},";
            else
                file << "}";
        }
        if(capa + 1 != neural_network.size())
            file << "},";
        else
            file << "}";
        capa++;
    }
    file << "}";
    

    file.close();

}

// Función para guardar en un fichero la red de una forma más organizada.
// Se usara este formato para construir la red a partir de un fichero.
void Neural_Network::save_network(int num_network,string filename){
    filename += to_string(num_network) + ".txt";
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "No se pudo abrir el archivo para guardar la red" << endl;
    }
    for(int i = 0;i<neural_network.size();i++){
        file << "Layer " << i << '\n';
        for(int j = 0;j < neural_network[i].size();j++){
            for(int k = 0;k < neural_network[i][j].size();k++){
                file << neural_network[i][j][k] << ' ';
            }
            file << '\n';
        }
        file << "Layer end" << '\n';
    }
}

void Neural_Network::read_network(string filename){
    neural_network.clear();
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "No se pudo abrir el archivo para leer la red" << endl;
    }
    string s;
    while(getline(file,s)){
        if(s.find("Layer") != string::npos){
            Matrix<double> layer;
            getline(file,s);
            string delimiters = " \n";
            while(s.find("Layer") == string::npos){
                vector<string> tokens = get_tokens(s,delimiters);
                vector<double> row;
                for(int i = 0;i < tokens.size();i++){
                    row.push_back(stod(tokens[i]));
                }
                layer.push_back(row);
                getline(file,s); 
            }
            neural_network.push_back(layer);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////