#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

const double PI = 3.141592653589793238462643383279502884;

struct Params {
    string input_dir;
    string output_dir;
    int num_configs = 100;
    double density = 1.0;
    double rbin_fraction = 0.005;
    int num_samples = 100000;
    int num_radii = 100;
    unsigned seed = 12345;
    double length_scale_a = 1.0;
};

int count_particles(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    int n = 0;
    double x, y;
    while (in >> x >> y) ++n;
    return n;
}

vector<pair<double,double>> read_config(const string& path, double L) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open " + path);
    vector<pair<double,double>> pts;
    double x, y;
    while (in >> x >> y) pts.push_back({x * L, y * L});
    return pts;
}

double pbc_abs(double dx, double L) {
    dx = fabs(dx);
    if (dx > 0.5 * L) dx = L - dx;
    return dx;
}

Params parse_args(int argc, char** argv) {
    if (argc < 10) {
        cerr << "usage: compute_number_variance input_dir output_dir num_configs density rbin_fraction num_samples num_radii seed length_scale_a\n";
        exit(1);
    }
    Params p;
    p.input_dir = argv[1];
    p.output_dir = argv[2];
    p.num_configs = stoi(argv[3]);
    p.density = stod(argv[4]);
    p.rbin_fraction = stod(argv[5]);
    p.num_samples = stoi(argv[6]);
    p.num_radii = stoi(argv[7]);
    p.seed = static_cast<unsigned>(stoul(argv[8]));
    p.length_scale_a = stod(argv[9]);
    return p;
}

int main(int argc, char** argv) {
    Params par = parse_args(argc, argv);
    string first = par.input_dir + "/config_0_component_0.txt";
    int N = count_particles(first);
    double L = sqrt(static_cast<double>(N) / par.density);
    double dr = par.rbin_fraction * L;

    vector<double> sigma(par.num_radii, 0.0);
    mt19937 gen(par.seed);
    uniform_real_distribution<double> uni(0.0, L);

    cout << "N=" << N << " L=" << L << " dr=" << dr << "\n";

    for (int c = 0; c < par.num_configs; ++c) {
        string path = par.input_dir + "/config_" + to_string(c) + "_component_0.txt";
        auto pts = read_config(path, L);
        if (static_cast<int>(pts.size()) != N) throw runtime_error("inconsistent N in " + path);
        cout << "number variance " << (c + 1) << "/" << par.num_configs << "\n";

        for (int r_idx = 0; r_idx < par.num_radii; ++r_idx) {
            double R = (r_idx + 1) * dr;
            double mean = 0.0, mean2 = 0.0;
            for (int s = 0; s < par.num_samples; ++s) {
                double cx = uni(gen);
                double cy = uni(gen);
                int nwin = 0;
                for (const auto& p : pts) {
                    double dx = pbc_abs(p.first - cx, L);
                    double dy = pbc_abs(p.second - cy, L);
                    if (dx * dx + dy * dy < R * R) ++nwin;
                }
                mean += nwin;
                mean2 += static_cast<double>(nwin) * nwin;
            }
            mean /= par.num_samples;
            mean2 /= par.num_samples;
            sigma[r_idx] += mean2 - mean * mean;
        }
    }

    ofstream out_R(par.output_dir + "/num_var_ensemble.txt");
    ofstream out_Ra(par.output_dir + "/num_var_ensemble_R_over_a.txt");
    for (int i = 0; i < par.num_radii; ++i) {
        double R = (i + 1) * dr;
        double val = sigma[i] / par.num_configs;
        out_R << R << "\t" << val << "\n";
        out_Ra << R / par.length_scale_a << "\t" << val << "\n";
    }
    return 0;
}
