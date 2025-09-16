#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>
#include <stdexcept>

using namespace std; 

class Bounds{
    private:
        int lower_bound, upper_bound;
    
    public:     
        Bounds(int lb, int ub) : lower_bound(lb), upper_bound(ub) {}

        static Bounds from_weights(const vector<int>& weights, int size) {
            int lb = 0;
            int ub = 0;
            int end_pt = min(size, static_cast<int>(weights.size()));

            for (int i = 0; i < end_pt; i++) {
                int weight = weights[i];
                if (weight < 0) {
                    lb += weight;
                }
                if (weight > 0) {
                    ub += weight;
                }
            }
            return Bounds(lb, ub);
        } 

        pair<int, int> get_bounds() const {
            return make_pair(lower_bound, upper_bound);
    }
};

class ThresholdTest{
    private:
        static int id_counter;
        int id;
        vector<int> weights;
        vector<int> indices;
        int threshold; 
        int size;
        Bounds bounds;

        // creates new id for object
        static int new_id() {
            return id_counter++;
        }
        
    public:
        ThresholdTest(const vector<int>& weights, int threshold,
                 const vector<int>& indices, int size, const Bounds& bounds)
            : weights(weights), threshold(threshold), indices(indices),
              size(size), bounds(bounds), id(ThresholdTest::new_id()) {}

        ThresholdTest(const vector<int>& weights, int threshold)
            : weights(weights), threshold(threshold), indices(),
              size(weights.size()), 
              bounds(Bounds::from_weights(weights, weights.size())),
              id(ThresholdTest::new_id()) {}

        // gets last weight from list 
        int get_last() const {
            return weights[size - 1];
        }

        shared_ptr<ThresholdTest> set_last(int value) const {
            int last_weight = get_last();
            int nu_threshold = threshold;

            if (value == 1) {
                nu_threshold = threshold - last_weight;
            }

            auto [lb, ub] = bounds.get_bounds();

            if (last_weight > 0) {
                ub -= last_weight;
            } else {
                lb -= last_weight;
            }
            Bounds nu_bounds(lb, ub);

            return make_shared<ThresholdTest>(weights, nu_threshold, indices, size - 1, nu_bounds);

        }

        static pair<vector<int>, vector<int>> sort_weights(const vector<int>& weights) {
            vector<pair<int, int>> indexed_weights;
            for (size_t i = 0; i < weights.size(); i++) {
                indexed_weights.emplace_back(i, weights[i]);
            }

            sort(indexed_weights.begin(), indexed_weights.end(),
                    [](const pair<int, int>& a, const pair<int, int>& b) {
                        return abs(a.second) < abs(b.second);
                    });

            vector<int> indices;
            vector<int> sorted_weights;
            for (const auto& pair : indexed_weights) {
                indices.push_back(pair.first);
                sorted_weights.push_back(pair.second);
            }

            return make_pair(indices, sorted_weights);
        }

        bool trivial_pass() const {
            auto [lower, upper] = bounds.get_bounds();
            return threshold <= lower && lower <= upper;
        }

        bool trivial_fail() const {
            auto [lower, upper] = bounds.get_bounds();
            return lower <= upper && upper < threshold;
        }

        int get_threshold() const { return threshold; }
};

int ThresholdTest::id_counter = 0;


int main() {
    vector<int> weights = {2, -1, 3};
    
    ThresholdTest test(weights, 2);
    cout << "original threshold:  " << test.get_threshold() << endl;

    return 0;
}

