/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 *
 * This program models a discrete-time process where adjacent pairs of a boy ('B')
 * followed by a girl ('G') swap their positions in a queue. This process is
 * repeated for a specified number of time steps.
 *
 * @b Algorithm: Direct simulation of local swaps.
 * @b Time_Complexity: O(T * N), where T is the number of time steps and N is the queue length.
 * @b Space_Complexity: O(N) to store the queue state.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t;
  string input;

  cin >> n >> t;
  cin >> input;
  vector<char> queue(input.begin(), input.end());

  /**
   * @brief Main simulation loop that advances time.
   * @invariant The loop runs for 't' iterations, each representing one second of time.
   */
  while (t > 0) {
    /**
     * @brief A single pass through the queue to perform swaps for the current time step.
     * @invariant This loop identifies and swaps all adjacent 'B'/'G' pairs in a single left-to-right scan.
     */
    for (int i = 1; i < queue.size(); i++) {
      if (queue[i] == 'G' && queue[i - 1] == 'B') {
        swap(queue[i], queue[i - 1]);
        /**
         * @brief Advance index to prevent a single element from moving multiple times in one pass.
         * This ensures that each swap is an atomic event within a single time step.
         */
        i++;
      }
    }
    t--;
  }

  // Output the final state of the queue after the simulation.
  for (char c : queue) {
    cout << c;
  }
  cout << '\n';
  return 0;
}
