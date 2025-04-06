#include <mpi.h>
#include <stdio.h>
#include "../include/csv.h"
#include <vector>
#include <string>

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
