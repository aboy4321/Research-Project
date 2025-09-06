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
        
        Bounds(int lb, int ub) : lower_bound(lb), upper_bound(ub) {}

        static Bounds from_weights(const vector<int>& weights, int size) {
            int lb, ub = 0, 0;
            
            int end_pt = min(size, static_cast<int>(weights.size()))

            for (int i = 0; i < range; i++) {
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
}

class ThresholdTest{
    private:
        int id_counter;
        vector<int> weights;
        int threshold; 
        int size;
        Bounds bounds;

        // creates new id for object
        static int new_id{
            return id_counter++;
        }
      
        ThresholdTest(const vector<int>& weights, int threshold)
            : weights(weights), threshold(threshold) {
            }

        // gets last weight from list 
        vector<int> get_last() const {
            return weights[size - 1];
        }

        shared_ptr<ThresholdTest> set_last(int value) const {
            int last_weight = get_last();
            double nu_threshold = threshold;

            if (value == 1) {
                int nu_threshold = threshold - last_weight;
            }
            if (value == 0) {
                int nu_threshold = threshold;
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
}


