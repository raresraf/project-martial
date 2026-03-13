/**
 * @file 266B.CPP
 * @brief Solution for CodeForces Problem 266B - "Queue at the School".
 * @author Yadi Qian
 * 
 * @details
 * This program simulates a queue of children, represented by 'B' for boys and 'G' for girls.
 * In each time step, any boy ('B') standing directly in front of a girl ('G') swaps places with her.
 * The simulation runs for a specified number of time steps (`t`).
 *
 * @algorithm Iterative in-place single-pass swap.
 * @time_complexity O(T * N), where T is the number of time steps and N is the length of the queue.
 * @space_complexity O(N) to store the queue string.
 *
 * @note The extensive list of typedefs and the `pi` constant are not used in this specific problem
 * but are common boilerplate in competitive programming templates.
 */
#include <bits/stdc++.h>

using namespace std;

// The following are common competitive programming type definitions.
// They are not used in this particular solution.
const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes 
   * to optimize high-throughput I/O.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;      // Length of the queue.
  int t;      // Number of time steps to simulate.
  string line; // The string representing the queue.
  cin >> n >> t >> line;
  
  /**
   * @brief Block that orchestrates the temporal progression of the simulation.
   * This outer loop runs the simulation for `t` discrete time steps.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @brief Block that performs a single-pass sweep to resolve 'BG' inversions.
     * This inner loop iterates through the queue to find adjacent 'B' and 'G' pairs
     * and swaps them in-place.
     */
    for (int j = 0; j < n - 1; j++) {
      // Pre-condition: Check if a boy is immediately followed by a girl.
      if (line[j] == 'B' && line[j + 1] == 'G') {
        // Perform the local state transition (swap).
        line[j] = 'G';
        line[j + 1] = 'B';
        /**
         * @brief Inline Logic: Skips the next index.
         * This is crucial to ensure that a single boy does not move more than one position
         * to the right in a single time step, preserving the atomic nature of the movement.
         */
        j++;
      }
    }
  }

  // Output the final state of the queue.
  cout << line;

  return 0;
}
