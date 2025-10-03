#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "windows.h"
#include <iomanip>
#include <fstream>
#include <string>


void Insertdata(const std::string& filename, const std::string& filenameTr, const std::vector<double>& inputs, const std::vector<double>& targets, const std::vector<double>& results) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл для записи данных!" << std::endl;
        return;
    }
    for (int i = 0; i < inputs.size(); i++) {
        file << inputs[i] << " " << results[i] << std::endl;
    }

    file.close();
    std::ofstream fileTr(filenameTr);
    if (!fileTr) {
        std::cerr << "Ошибка: не удалось открыть файл для записи данных!" << std::endl;
        return;
    }
    for (int i = 0; i < inputs.size(); i++) {
        fileTr << inputs[i] << " " << targets[i] << std::endl;
    }

    fileTr.close();
};

int visualData(const std::vector<double>& inputs, const std::vector<double>& targets, const std::vector<double>& results) {
    const std::string dataFile = "data.txt";
    const std::string dataTrue = "dataTrue.txt";
    const std::string plotScript = "plot_script.gnuplot";
    Insertdata(dataFile, dataTrue, inputs, targets, results);

    // Создание скрипта для Gnuplot
    std::ofstream script(plotScript);
    if (!script) {
        std::cerr << "Ошибка: не удалось создать файл скрипта для Gnuplot!" << std::endl;
        return 1;
    }

    script << "set title '2D Plot'\n";
    script << "set xlabel 'X-axis'\n";
    script << "set ylabel 'Y-axis'\n";
    script << "set grid\n";
    script << "set terminal wxt size 1600,900\n";
    script << "set yrange [-3:3]\n";
    script << "set xrange [-7:7]\n";
    script << "plot '" << dataTrue << "' using 1:2 with lines lc rgb \"red\" title 'data', '" << dataFile << "' using 1:2 with lines lc rgb \"green\" title 'data'\n";
    script.close();

    // Вызов Gnuplot
    std::string command = "gnuplot -persist " + plotScript;
    int result = system(command.c_str());
    if (result != 0) {
        std::cerr << "Ошибка: не удалось запустить Gnuplot!" << std::endl;
    }

    return 0;
};

// Функция активации (сигмоида)
double sigmoid(double x) {
    // return 1.0 / (1.0 + exp(-0.5*x));

    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
};

// Производная сигмоиды
double sigmoid_derivative(double x) {
    // double s = sigmoid(x);
    // return s * (1.0 - s);
    return (1.0 - pow(sigmoid(x), 2));
};

double error_count(double output, double target) {
    double res =  pow(output - target, 2) * 0.5;
    // std::cout << output << " " << target << " " << res  << std::endl;
    return res;
};

// Структура для хранения слоя нейронов
struct Layer {
    int neurons;
    std::vector<std::vector<double>> weights;

    void LayerInit(int num_neurons){
        neurons = num_neurons;
        weights.resize(num_neurons);
        // Инициализация весов случайными значениями
        std::srand(std::time(0));
        for (int i = 0; i < neurons; i++) {
                weights[i].resize(3);
                weights[i][0] = ((double)std::rand() / RAND_MAX * 4 - 2);
                weights[i][1] = ((double)std::rand() / RAND_MAX * 4 - 2);
                weights[i][2] = ((double)std::rand() / RAND_MAX * 2 - 1);
        }
        
    }
};

// Класс многослойного персептрона
class MLP {
private:
    int layersCount;
    Layer layer;

public:
    MLP(const std::vector<int>& topology) {
        layer.LayerInit(topology[1]);
        std::cout << "neurons: " << topology[1] << " " << layer.neurons << std::endl;
        getWeights();
        std::cout<< std::endl; 
    };

    void getWeights(){
        for (int j = 0; j < layer.neurons; ++j) {
            std::cout << "  neuron " << j;
            std::cout << ": w1" << " - " << layer.weights[j][0];
            std::cout << ", w2" << " - " << layer.weights[j][1];
            std::cout << ", w3" << " - " << layer.weights[j][2] << std::endl; 
        }
    };

    double forward(double input) {
        std::vector<double> activations;
        double sum = 0;
        activations.resize(layer.neurons);
        for (int i = 0; i < layer.neurons; ++i) {
            sum = 0;
            sum += input * layer.weights[i][0];
            sum += 1 * layer.weights[i][1];
            activations[i] = sigmoid(sum);
        };
        sum = 0;
        for (int i = 0; i < layer.neurons; ++i){
            sum += activations[i] * layer.weights[i][2];
        };
        return sum;
    };

    // Обучение
    void gradTrain(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        std::vector<std::vector<double>> deltas;
        std::vector<std::vector<double>> prevWeights;
        std::vector<double> activationsFirst(layer.neurons);
        std::vector<double> activationsSecond(layer.neurons);
        double activationsSecondSum;

        std::vector<double> results;
        int learnPos;
        double u = 0, error, errorNext, curError, i, res, eps = 1, epsMin;
        int q = 0;
        while (q < 1000000){
            q++;
            i = ((double)std::rand() / RAND_MAX * 14 - 7);
            res = sin(i)*cos(i);
            activationsSecondSum = 0;
            deltas.clear();
            deltas.resize(3);
            error = 0;
            errorNext = 0;
            error = error_count(forward(i), res);
            for (int j = 0; j < layer.neurons; ++j) {
                u = i * layer.weights[j][0];
                u += 1 * layer.weights[j][1];
                activationsFirst[j] = u;
                activationsSecond[j] = sigmoid(u);
                activationsSecondSum += activationsSecond[j] * layer.weights[j][2];
            } 

            for (int j = 0; j < layer.neurons; ++j) {
                double delta = 0, xDelta = 0;;
                delta = (forward(i) - res) * activationsSecondSum;
                deltas[2].push_back(delta * activationsSecond[j]);
                delta = delta * layer.weights[j][2] * sigmoid_derivative(activationsFirst[j]);
                deltas[1].push_back(delta);
                deltas[0].push_back(delta * i);
                
            }
            prevWeights = layer.weights;
            eps = 1;
            epsMin = eps;
            curError = error;
            while(eps > learning_rate){
                // Обновление новых весов         
                for (int j = 0; j < layer.neurons; ++j) {
                    layer.weights[j][0] -= eps * deltas[0][j];
                    layer.weights[j][1] -= eps * deltas[1][j];
                    layer.weights[j][2] -= eps * deltas[2][j];
                    // std::cout << j << " w0: " << layer.weights[j][0] << ", w1: " << layer.weights[j][1] << ", w2: " << layer.weights[j][2] << std::endl;
                    // std::cout << j << " d0: " << deltas[0][j] << ", d1: " << deltas[1][j] << ", d2: " << deltas[2][j] << std::endl;
                    // std::cout << std::endl;
                }

                errorNext = error_count(forward(i), res);
                if (errorNext < error) {
                    error = errorNext;
                    epsMin = eps;
                } 
                eps = eps/2;
                layer.weights = prevWeights;
                // std::cout << eps << " " << errorNext << " " << error << " " << epsMin << " " << i << " " << res << " " << forward(i)<< std::endl;

            }
            eps = -1;
            while(eps < -1*learning_rate){
                // Обновление новых весов         
                for (int j = 0; j < layer.neurons; ++j) {
                    layer.weights[j][0] -= eps * deltas[0][j];
                    layer.weights[j][1] -= eps * deltas[1][j];
                    layer.weights[j][2] -= eps * deltas[2][j];
                    // std::cout << j << " w0: " << layer.weights[j][0] << ", w1: " << layer.weights[j][1] << ", w2: " << layer.weights[j][2] << std::endl;
                    // std::cout << j << " d0: " << deltas[0][j] << ", d1: " << deltas[1][j] << ", d2: " << deltas[2][j] << std::endl;
                    // std::cout << std::endl;
                }

                errorNext = error_count(forward(i), res);
                if (errorNext < error) {
                    error = errorNext;
                    epsMin = eps;
                } 
                eps = eps/2;
                layer.weights = prevWeights;
                // std::cout << eps << " " << errorNext << " " << error << " " << epsMin << " " << i << " " << res << " " << forward(i)<< std::endl;

            }
            // std::cout << std::endl;
            if (error < curError) {
                for (int j = 0; j < layer.neurons; ++j) {
                    layer.weights[j][0] -= epsMin * deltas[0][j];
                    layer.weights[j][1] -= epsMin * deltas[1][j];
                    layer.weights[j][2] -= epsMin * deltas[2][j];  
            }}
            // std::cout << " iter: " << i << ", lp: " << epsMin << ", err: " << error << ", errnext: " << curError << std::endl;
            if (q % 10000 == 0) {
                // getWeights();
                std::cout << " iter: " << q << std::endl;
            }
            if (q % 100000 == 0) {
                getWeights();
                for (int k = 0; k < input.size(); k++) { 
                    results.push_back(forward(input[k]));
                }
                int d = visualData(input, target, results);
                results.clear();
            }
        }
        getWeights();
    };
};

int main() {
    MLP mlp({1, 15}); // Топология: 1 скрытый слой, 15 нейронов
    // Генерация данных для обучения
    std::vector<double> inputs;
    std::vector<double> targets;
    std::vector<double> results;
    double xn, x;
    for (double xn = -6.4; xn <= 6.4; xn += 0.05) {
        x = xn;
        inputs.push_back(x);
        // targets.push_back(sin(x));
        targets.push_back(sin(x)*cos(x));
    };

    for (int i = 0; i < inputs.size(); i++) { 
        results.push_back(mlp.forward(inputs[i]));
    }
    int d = visualData(inputs, targets, results);

    // Обучение
    mlp.gradTrain(inputs, targets, 0.0001);

    // Тестирование
    // for (int i = 0; i < inputs.size(); i++) { 
    //     results.push_back(mlp.forward(inputs[i]));
    // }
    // int d = visualData(inputs, targets, results);
    return 0;
}
