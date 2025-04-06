#include "../include/NeuralNetwork.h"
#include <Eigen/Dense>
#include <cmath>

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
    std::vector<int> predict(const Eigen::MatrixXd& X) {
        std::vector<int> preds;
        preds.reserve(X.cols());
        for (int i = 0; i < X.cols(); ++i) {
            preds.push_back(predict_one(X.col(i)));
        }
        return preds;
    }
    // Обучение на модела върху даден тренировъчен набор
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& y, int epochs, double lr) {
        int n = X.cols();
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            for (int i = 0; i < n; ++i) {
                // 1. Forward pass за пример i
                Eigen::VectorXd x = X.col(i);
                double t = (double)y[i];              // целева стойност (0 или 1)
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
};
