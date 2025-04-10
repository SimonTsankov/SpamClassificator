
#include "Eigen/Dense"
#include <cmath>
#include <vector>
#include <fstream>
class NeuralNetwork {
private:
    // Параметри на модела: тегла и bias за 2-слойна мрежа (един скрит слой)
    Eigen::MatrixXd W1; Eigen::VectorXd b1;  // тегла и bias за скрития слой
    Eigen::MatrixXd W2; Eigen::VectorXd b2;  // тегла и bias за изходния слой
    int input_dim, hidden_dim;
public:
    NeuralNetwork(int input_dim, int hidden_dim) : input_dim(input_dim), hidden_dim(hidden_dim) {
        // Инициализация на теглата (Random за W, 0 за bias)
        W1 = Eigen::MatrixXd::Random(hidden_dim, input_dim);
        b1 = Eigen::VectorXd::Zero(hidden_dim);
        W2 = Eigen::MatrixXd::Random(1, hidden_dim);
        b2 = Eigen::VectorXd::Zero(1);
    }
    // Предно разпространение: връща изходен вектор (вероятност(и))
    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        // Линеен комбинатор след скрития слой + tanh активация
        Eigen::VectorXd h = (W1 * x + b1).unaryExpr([](double z) { return tanh(z); });
        // Изходен неврон със сигмоид
        Eigen::VectorXd o = W2 * h + b2;
        return o.unaryExpr([](double z) { return 1.0 / (1.0 + exp(-z)); });
    }
    // Предсказание за единичен пример: 0 или 1 в зависимост от сигмоидния изход
    int predict_one(const Eigen::VectorXd& x) {
        double prob = forward(x)[0];
        return (prob >= 0.5 ? 1 : 0);
    }
    // Предсказания за множество примери (колони на матрицата X)
    Eigen::VectorXi predict(const Eigen::MatrixXd& X) {
        int n_samples = X.rows();  // приемаме, че семплите са редове
        Eigen::VectorXi predictions(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            Eigen::VectorXd x = X.row(i).transpose();  // транспонираме ред до вектор-колона
            predictions(i) = predict_one(x);
        }

        return predictions;
    }
    // Обучение на модела върху даден тренировъчен набор
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& y, int epochs, double lr) {
        int n = X.cols();
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            for (int i = 0; i < n; ++i) {
                // 1. Forward pass за пример i
                Eigen::VectorXd x = X.col(i);
                double t = (double)y[i];
                // целева стойност (0 или 1)
                Eigen::VectorXd h = (W1 * x + b1).unaryExpr([](double z) { return tanh(z); });
                double o = (W2 * h + b2)[0];
                double pred = 1.0 / (1.0 + exp(-o));  // сигмоидна прогноза

                // 2. Изчисление на грешката и градиентите (backpropagation)
                double error = pred - t;              // разлика между прогноза и цел
                // Градиенти за изходния слой:
                Eigen::RowVectorXd gradW2 = error * h.transpose();  // (1xhidden) градиент за W2
                double gradb2 = error;
                // Градиенти за скрития слой:
                Eigen::VectorXd dh = W2.transpose() * error;       // грешка, пренесена към скритите неврони
                Eigen::VectorXd grad_h = (1 - h.array().square()).matrix().cwiseProduct(dh);
                Eigen::MatrixXd gradW1 = grad_h * x.transpose();    // (hidden x input) градиент за W1
                Eigen::VectorXd gradb1 = grad_h;
                // 3. Ъпдейт на параметрите с градиентен спад
                W2 -= lr * gradW2;
                b2 -= lr * Eigen::VectorXd::Constant(1, gradb2);
                W1 -= lr * gradW1;
                b1 -= lr * gradb1;
            }
        }
    }

    // Запазване на модела във файл
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for saving model.");

        auto write_matrix = [&](const Eigen::MatrixXd& mat) {
            int rows = mat.rows(), cols = mat.cols();
            out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            out.write(reinterpret_cast<const char*>(mat.data()), sizeof(double) * rows * cols);
        };

        auto write_vector = [&](const Eigen::VectorXd& vec) {
            int size = vec.size();
            out.write(reinterpret_cast<const char*>(&size), sizeof(int));
            out.write(reinterpret_cast<const char*>(vec.data()), sizeof(double) * size);
        };

        write_matrix(W1);
        write_vector(b1);
        write_matrix(W2);
        write_vector(b2);
        out.close();
    }

    // Зареждане на модела от файл
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for loading model.");

        auto read_matrix = [&](Eigen::MatrixXd& mat) {
            int rows, cols;
            in.read(reinterpret_cast<char*>(&rows), sizeof(int));
            in.read(reinterpret_cast<char*>(&cols), sizeof(int));
            mat.resize(rows, cols);
            in.read(reinterpret_cast<char*>(mat.data()), sizeof(double) * rows * cols);
        };

        auto read_vector = [&](Eigen::VectorXd& vec) {
            int size;
            in.read(reinterpret_cast<char*>(&size), sizeof(int));
            vec.resize(size);
            in.read(reinterpret_cast<char*>(vec.data()), sizeof(double) * size);
        };

        read_matrix(W1);
        read_vector(b1);
        read_matrix(W2);
        read_vector(b2);
        in.close();
    }
};
