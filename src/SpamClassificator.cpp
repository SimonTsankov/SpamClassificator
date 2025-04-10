#include <mpi.h>
#include <stdio.h>
#include "../include/csv.h"
//#include <vector>
#include <string>
//#include <mpi.h>
#include "../include/Eigen/Dense"
#include "../include/NeuralNetwork.h"
//#include "utils.h"
#include <iostream>
#include <vector>
//#include <random>
//#include <ctime>

#include <algorithm>
#include <random>
#include <ctime>
#include <cctype>
#include <sstream>
#include <unordered_map>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

void load_csv(const std::string& filename,
    std::vector<std::string>& texts, std::vector<int>& labels) {
   
    io::CSVReader<2, io::trim_chars<' '>, io::double_quote_escape<',', '\"'>> csv(filename);
    csv.read_header(io::ignore_extra_column, "text", "target");
   
    std::string email_text;
    int target;
    while (csv.read_row(email_text, target)) {
        texts.push_back(email_text);
        labels.push_back(target);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Зареждане на CSV файла с имейли и етикети (колони "text" и "target")
    std::vector<std::string> texts;
    std::vector<int> labels;
    if (rank == 0) {
        load_csv("C:/Users/vladi/OneDrive/Desktop/Parallel data processing/SpamClassificator/data/short.csv", texts, labels);
    }

    // 2. Разбъркване на индексите и разделяне на данните на обучаващи (train) и тестови
    int total_samples;
    if (rank == 0) {
        total_samples = texts.size();
    }
    MPI_Bcast(&total_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Създаване на масив indices = [0,1,...,N-1] и разбъркването му в ранг 0
    std::vector<int> indices;
    if (rank == 0) {
        indices.resize(total_samples);
        for (int i = 0; i < total_samples; ++i)
            indices[i] = i;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
    else {
        indices.resize(total_samples);
    }
    MPI_Bcast(indices.data(), total_samples, MPI_INT, 0, MPI_COMM_WORLD);
    // Определяне на брой training и test семпли (80% train, 20% test)
    int train_count = 0;
    int test_count = 0;
    if (rank == 0) {
        train_count = static_cast<int>(0.8 * total_samples);
        if (train_count < 1) train_count = total_samples - 1;
        test_count = total_samples - train_count;
    }
    MPI_Bcast(&train_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Извличане на списъците с train и test данни в ранг 0 според разбърканите индекси
    std::vector<std::string> train_texts;
    std::vector<int> train_labels;
    std::vector<std::string> test_texts;
    std::vector<int> test_labels;
    if (rank == 0) {
        train_texts.reserve(train_count);
        train_labels.reserve(train_count);
        test_texts.reserve(test_count);
        test_labels.reserve(test_count);
        for (int i = 0; i < train_count; ++i) {
            train_texts.push_back(texts[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        }
        for (int j = 0; j < test_count; ++j) {
            test_texts.push_back(texts[indices[train_count + j]]);
            test_labels.push_back(labels[indices[train_count + j]]);
        }
    }

    // 3. Създаване на речник (уникални думи) от тренировъчните текстове на процес 0
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int> word_to_index;
    // Помощна Lambda-функция за токенизация: малки букви и разделяне по интервали
    auto tokenize = [&](const std::string& text) {
        std::string cleaned = text;
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        for (char& ch : cleaned) {
            if (!std::isalnum(static_cast<unsigned char>(ch))) {
                ch = ' ';
            }
        }
        std::istringstream iss(cleaned);
        std::string token;
        std::vector<std::string> tokens;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    };
    if (rank == 0) {
        vocabulary.reserve(10000);
        for (const std::string& text : train_texts) {
            std::vector<std::string> tokens = tokenize(text);
            for (const std::string& word : tokens) {
                if (word_to_index.find(word) == word_to_index.end()) {
                    // добавяне на нова дума в речника
                    int index = vocabulary.size();
                    vocabulary.push_back(word);
                    word_to_index[word] = index;
                }
            }
        }
    }

    // 4. Изпращане на речника към всички процеси (чрез MPI_Bcast)
    int vocab_size = 0;
    int hidden_dim = 16;
    int epochs = 2;
    double lr = 0.01;
    if (rank == 0) vocab_size = vocabulary.size();
    MPI_Bcast(&vocab_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Изпращане на всички думи като един единствен стринг, разделен с '\n'
    std::string vocab_buffer;
    int vocab_chars = 0;
    if (rank == 0) {
        for (const std::string& w : vocabulary) {
            vocab_chars += w.size() + 1;  // броим символите + разделител
        }
        vocab_buffer.reserve(vocab_chars);
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            vocab_buffer += vocabulary[i];
            if (i < vocabulary.size() - 1) vocab_buffer += '\n';
        }
    }
    MPI_Bcast(&vocab_chars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        vocab_buffer.resize(vocab_chars);
    }
    MPI_Bcast(&vocab_buffer[0], vocab_chars, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Реконструиране на списъка vocabulary от получения стринг
        vocabulary.clear();
        vocabulary.reserve(vocab_size);
        std::istringstream iss(vocab_buffer);
        std::string word;
        while (std::getline(iss, word, '\n')) {
            vocabulary.push_back(word);
        }
        word_to_index.clear();
        for (int i = 0; i < vocab_size; ++i) {
            word_to_index[vocabulary[i]] = i;
        }
    }

    // 5. Изпращане на тренировъчните и тестови данни към всички процеси
    //    (етикети и текстове на имейлите)
    if (rank != 0) {
        train_labels.resize(train_count);
        test_labels.resize(test_count);
    }
    MPI_Bcast(train_labels.data(), train_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_labels.data(), test_count, MPI_INT, 0, MPI_COMM_WORLD);
    // Подготовка и изпращане на тренировъчните текстове
    std::vector<int> train_text_lengths;
    std::string train_text_buffer;
    int train_text_chars = 0;
    if (rank == 0) {
        train_text_lengths.resize(train_count);
        for (int i = 0; i < train_count; ++i) {
            train_text_lengths[i] = train_texts[i].size();
            train_text_chars += train_text_lengths[i];
        }
        train_text_buffer.reserve(train_text_chars);
        for (const std::string& t : train_texts) {
            train_text_buffer += t;
        }
    }
    else {
        train_text_lengths.resize(train_count);
    }
    MPI_Bcast(train_text_lengths.data(), train_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&train_text_chars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        train_text_buffer.resize(train_text_chars);
    }
    MPI_Bcast(&train_text_buffer[0], train_text_chars, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Реконструиране на вектора train_texts
        train_texts.clear();
        train_texts.reserve(train_count);
        int offset = 0;
        for (int i = 0; i < train_count; ++i) {
            std::string t = train_text_buffer.substr(offset, train_text_lengths[i]);
            train_texts.push_back(t);
            offset += train_text_lengths[i];
        }
    }
    // Подготовка и изпращане на тестовите текстове по същия начин
    std::vector<int> test_text_lengths;
    std::string test_text_buffer;
    int test_text_chars = 0;
    if (rank == 0) {
        test_text_lengths.resize(test_count);
        for (int j = 0; j < test_count; ++j) {
            test_text_lengths[j] = test_texts[j].size();
            test_text_chars += test_text_lengths[j];
        }
        test_text_buffer.reserve(test_text_chars);
        for (const std::string& t : test_texts) {
            test_text_buffer += t;
        }
    }
    else {
        test_text_lengths.resize(test_count);
    }
    MPI_Bcast(test_text_lengths.data(), test_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_text_chars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        test_text_buffer.resize(test_text_chars);
    }
    MPI_Bcast(&test_text_buffer[0], test_text_chars, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Реконструиране на вектора test_texts
        test_texts.clear();
        test_texts.reserve(test_count);
        int offset = 0;
        for (int j = 0; j < test_count; ++j) {
            std::string t = test_text_buffer.substr(offset, test_text_lengths[j]);
            test_texts.push_back(t);
            offset += test_text_lengths[j];
        }
    }

    // 6. Bootstrap семплиране на тренировъчните данни във всеки процес
    std::mt19937 rng(static_cast<unsigned>(time(NULL)) + rank);
    std::uniform_int_distribution<int> dist(0, train_count - 1);
    std::vector<int> sample_indices(train_count);
    for (int i = 0; i < train_count; ++i) {
        sample_indices[i] = dist(rng);
    }

    // 7. Векторизация на тренировъчните имейли и обучение на локалния модел
    Eigen::MatrixXd X_train(train_count, vocab_size);
    X_train.setZero();
    Eigen::VectorXi y_train(train_count);
    for (int i = 0; i < train_count; ++i) {
        int idx = sample_indices[i];
        y_train(i) = train_labels[idx];
        // Преобразуване на текста в честотен вектор чрез речника
        std::vector<std::string> tokens = tokenize(train_texts[idx]);
        for (const std::string& w : tokens) {
            auto it = word_to_index.find(w);
            if (it != word_to_index.end()) {
                int j = it->second;
                X_train(i, j) += 1.0f;
            }
        }
    }
    NeuralNetwork model(vocab_size, hidden_dim);
    model.train(X_train.transpose(), y_train, epochs, lr);

    // 8. Векторизация на тестовите имейли и предсказване с обучените модели
    Eigen::MatrixXd X_test(test_count, vocab_size);
    X_test.setZero();
    for (int j = 0; j < test_count; ++j) {
        std::vector<std::string> tokens = tokenize(test_texts[j]);
        for (const std::string& w : tokens) {
            auto it = word_to_index.find(w);
            if (it != word_to_index.end()) {
                int col = it->second;
                X_test(j, col) += 1.0f;
            }
        }
    }
    Eigen::VectorXi local_preds = model.predict(X_test);

    // 9. Събиране на предсказанията от всички процеси и гласуване на мнозинството (hard voting)
    std::vector<int> local_pred_vec(test_count);
    for (int j = 0; j < test_count; ++j) {
        local_pred_vec[j] = local_preds(j);
    }
    std::vector<int> vote_sum(test_count);
    MPI_Reduce(local_pred_vec.data(), vote_sum.data(), test_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int correct = 0;
        for (int j = 0; j < test_count; ++j) {
            // Ако повече от половината модели дават 1 (спам), приемаме, че ансамбълът предсказва 1
            int ensemble_pred = (vote_sum[j] > size / 2) ? 1 : 0;
            if (ensemble_pred == test_labels[j]) {
                correct++;
            }
        }
        double accuracy = (double)correct / test_count;
        std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;
    }

    // 10. Записване на обучен модел във файл
    fs::create_directories("models");
    std::string model_filename = "models/model_rank_" + std::to_string(rank) + ".nn";
    model.save(model_filename);
    if (rank == 0) {
        std::cout << "Моделите са запазени в директорията 'models/'" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
    /*
    int rank, size;

    MPI_Init(&argc, &argv);             // инициализация на MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер на процеса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // брой процеси

    printf("Hello from process %d of %d\n", rank, size);

    std::vector<std::string> texts;
    std::vector<int> labels;
    load_csv("data/spam_assassin.csv", texts, labels);
    printf("Loaded %d", texts.size());
    MPI_Finalize();                     // завършване на MPI
    return 0;
}
*/