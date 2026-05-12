#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const double PI = 3.141592653589793238462643383279502884;

struct Params {
    string input_dir;
    string output_dir;
    int num_configs = 100;
    double density = 1.0;
    int nk = 200;
    double kbin_factor = 1.0;
    int max_bins = 1000;
    double length_scale_a = 1.0;
};

vector<pair<double,double>> read_config(const string& path, double L) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    vector<pair<double,double>> pts;
    double x, y;
    while (in >> x >> y) pts.push_back({x * L, y * L});
    return pts;
}

int count_particles(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    int n = 0;
    double x, y;
    while (in >> x >> y) ++n;
    return n;
}

double compute_sk(const vector<pair<double,double>>& pts, double kx, double ky) {
    double c = 0.0, s = 0.0;
    for (const auto& p : pts) {
        double q = p.first * kx + p.second * ky;
        c += cos(q);
        s += sin(q);
    }
    return (c * c + s * s) / static_cast<double>(pts.size());
}

Params parse_args(int argc, char** argv) {
    if (argc < 9) {
        cerr << "usage: compute_sk input_dir output_dir num_configs density nk kbin_factor max_bins length_scale_a\n";
        exit(1);
    }
    Params p;
    p.input_dir = argv[1];
    p.output_dir = argv[2];
    p.num_configs = stoi(argv[3]);
    p.density = stod(argv[4]);
    p.nk = stoi(argv[5]);
    p.kbin_factor = stod(argv[6]);
    p.max_bins = stoi(argv[7]);
    p.length_scale_a = stod(argv[8]);
    return p;
}

int main(int argc, char** argv) {
    Params par = parse_args(argc, argv);
    string first = par.input_dir + "/config_0_component_0.txt";
    int N = count_particles(first);
    double L = sqrt(static_cast<double>(N) / par.density);
    double dk = (2.0 * PI / L) * par.kbin_factor;

    vector<double> hist(par.max_bins, 0.0);
    vector<long long> count(par.max_bins, 0);

    cout << "N=" << N << " L=" << L << " dk=" << dk << "\n";

    for (int c = 0; c < par.num_configs; ++c) {
        string path = par.input_dir + "/config_" + to_string(c) + "_component_0.txt";
        auto pts = read_config(path, L);
        if (static_cast<int>(pts.size()) != N) throw runtime_error("inconsistent N in " + path);
        cout << "S(k) " << (c + 1) << "/" << par.num_configs << "\n";

        for (int i = 1; i <= par.nk; ++i) {
            for (int j = -par.nk; j <= par.nk; ++j) {
                double kx = i * 2.0 * PI / L;
                double ky = j * 2.0 * PI / L;
                double k = sqrt(kx * kx + ky * ky);
                int b = static_cast<int>(floor(k / dk));
                if (b >= 0 && b < par.max_bins) {
                    hist[b] += compute_sk(pts, kx, ky);
                    count[b] += 1;
                }
            }
        }
        for (int j = 1; j <= par.nk; ++j) {
            double kx = 0.0;
            double ky = j * 2.0 * PI / L;
            double k = fabs(ky);
            int b = static_cast<int>(floor(k / dk));
            if (b >= 0 && b < par.max_bins) {
                hist[b] += compute_sk(pts, kx, ky);
                count[b] += 1;
            }
        }
    }

    ofstream out_k(par.output_dir + "/SK_ensemble.txt");
    ofstream out_ka(par.output_dir + "/SK_ensemble_ka.txt");
    for (int b = 0; b < par.max_bins; ++b) {
        if (count[b] == 0) continue;
        double k = (b + 0.5) * dk;
        double sk = hist[b] / static_cast<double>(count[b]);
        out_k << k << "\t" << sk << "\n";
        out_ka << k * par.length_scale_a << "\t" << sk << "\n";
    }
    return 0;
}
