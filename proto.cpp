#include <iostream>
#include <vector>
#include <memory>

using namespace std; 

class Bounds{
    private:
        int lb, ub;
        
        Bounds();

        static pair<int, int> from_weights 


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
            
            if(value == 1) {
                int nu_threshold = threshold - last_weight;
            }
            if(value == 0) {
                int nu_threshold = threshold;
            }

        }
}


