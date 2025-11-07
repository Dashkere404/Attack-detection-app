#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
extern "C" {
#include "random_forest_model.h"
}

using namespace std;

vector<vector<double>> read_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;

        while (getline(ss, value, ',')) {
            // Remove leading/trailing spaces
            size_t start = value.find_first_not_of(" \t\r\n");
            size_t end = value.find_last_not_of(" \t\r\n");
            if (start != string::npos) {
                value = value.substr(start, end - start + 1);
            }

            try {
                double val = stod(value);
                row.push_back(val);
            }
            catch (const exception&) {
                throw; // Re-throw to stop processing
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    return data;
}

string remove_quotes(const string& s) {
    size_t start = 0;
    size_t end = s.length();

    if (end > 0 && s[0] == '"') start++;
    if (end > start && s[end - 1] == '"') end--;

    return s.substr(start, end - start);
}

int main() {
    string path_samples, path_true_samples;
    cout << "Enter path to sample CSV (without labels): ";
    getline(cin, path_samples);
    string path_s = remove_quotes(path_samples);

    cout << "Enter path to labels CSV (with true labels): ";
    getline(cin, path_true_samples);
    string path_ts = remove_quotes(path_true_samples);
    const char* labels[] = { "Benign", "DDoS", "DoS", "PortScan" };
    int n_classes = sizeof(labels) / sizeof(labels[0]);

    try {
        auto samples = read_csv(path_s);
        auto true_sample = read_csv(path_ts);
        cout << "Loaded " << samples.size() << " examples.\n\n";

        for (size_t i = 0; i < samples.size(); ++i) {
            auto true_row = true_sample[i];
            int true_label = true_row.back();

            const auto& features = samples[i];
            double output[4];
            vector<double> mutable_features = features;

            predict_risk(mutable_features.data(), output);

            // Find the class with the highest probability


            int pred_class = 0;
            double max_prob = output[0];
            for (int j = 1; j < n_classes; ++j) {
                if (output[j] > max_prob) {
                    max_prob = output[j];
                    pred_class = j;
                }
            }
            bool match = (pred_class == (int)true_label);
            cout << "Example " << (i + 1) << ": " << "Predicted: " << labels[pred_class] << " | True: " << labels[true_label] << " | Match: " << (match ? "True" : "False") << endl;
        }

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}