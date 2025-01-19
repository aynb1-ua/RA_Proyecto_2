#include <iostream>
#include <cmath>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <SDL/SDL.h>
#include <string>
#include <fstream>
#include "src/ale_interface.hpp"
#include "src/neural_network.cpp"

using namespace std;

// Constants
constexpr uint32_t maxSteps = 7500;

uint32_t seed = static_cast<uint32_t>(std::time(0));

template<typename T>
void push_data(vector<Matrix<T>> &data_points,ALEInterface &alei){
    Matrix<double> data;
    for(int i = 0; i < 128;i++){
        data.push_back({(unsigned int)alei.getRAM().get(i)});
    }
    data_points.push_back(data);
}


void save_data(const vector<Matrix<double>> &data_points,const vector<double> &output){
    ofstream file;
    file.open("data.csv");
    if(file){
        for(int i = 0; i < data_points.size();i++){
            for(int j = 0; j < data_points[i].size();j++){
                file << data_points[i][j][0] << ",";
            }
            file << output[i] << '\n';
        }
    }
    file.close();
}


///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////

vector<unsigned int> imprimir_ram(ALEInterface &alei,const vector<unsigned int> &ram){
    vector<unsigned int> new_ram;
    for(int i = 0;i < 128;i++){
        new_ram.push_back((unsigned int)alei.getRAM().get(i));
        //Imprime el valor de la ram obtenido en rojo al haber cambiado.
        if(new_ram[i] != ram[i]){
            cout << "\033[" << 31 << "m" << new_ram[i] << ' '; 
        }
        //Imprime el valor de la ram obtenido en verde al no haber cambiado.
        else{
            cout << "\033["<< 32 << "m" << new_ram[i] << ' '; 
        }
    }
    cout << endl;
    return new_ram;
}

void process_output(Action output, double &data_output){
    if(output == PLAYER_A_RIGHT){data_output = -0.75;}
    else if(output == PLAYER_A_LEFT){data_output = -0.25;}
    else if(output == PLAYER_A_FIRE){data_output = 0.25;}
    else{data_output = 0.75;}
}
int32_t getPlayerX(ALEInterface& alei) {
   return alei.getRAM().get(72) + ((rand() % 3) - 1);
}

int32_t getBallX(ALEInterface& alei) {
   return alei.getRAM().get(99) + ((rand() % 3) - 1);
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////



reward_t agentStep(ALEInterface& alei,double &data_output) {
   static constexpr int32_t wide { 9 };
   static int32_t lives { alei.lives() };
   reward_t reward{0};

   // When we loose a live, we need to press FIRE to start again
   if (alei.lives() < lives) {
      lives = alei.lives();
      alei.act(PLAYER_A_FIRE);
   }
   // Apply rules.
   auto playerX { getPlayerX(alei) };
   auto ballX   { getBallX(alei)   };
   Action output;
   uInt8 *keystate = SDL_GetKeyState(NULL);
   if       (keystate[SDLK_LEFT]) { output = PLAYER_A_LEFT;  }
   else if  (keystate[SDLK_RIGHT]) { output = PLAYER_A_RIGHT;  }
   else if (keystate[SDLK_UP]){output = PLAYER_A_FIRE; }
   else {output = PLAYER_A_NOOP;}
   reward = alei.act(output);
   process_output(output,data_output);
   return reward;
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>\n";
   exit(-1);
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   reward_t totalReward{};
   ALEInterface alei{};

   vector<Matrix<double>> neural_network;
   
   // Check input parameter
   if (argc != 2)
      usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", true);
   alei.loadROM (argv[1]);

   // Init
   std::srand(seed);
   vector<Matrix<double>> data_points;
   vector<double> outputs;
   // Main loop
   {
      alei.act(PLAYER_A_FIRE);
      uint32_t step{};
      vector<unsigned int> ram(128,0);
      while ( !alei.game_over() && step < maxSteps ) { 
        double data_output = 0.0;
        totalReward += agentStep(alei,data_output);
        ++step; 
        ram = imprimir_ram(alei,ram);
        outputs.push_back(data_output);
        push_data<double>(data_points,alei);
      }
      save_data(data_points,outputs);  
      std::cout << "Steps: " << step << std::endl;
      std::cout << "Reward: " << totalReward << std::endl;
      cout << data_points.size() << endl;
   }

   return 0;
}
