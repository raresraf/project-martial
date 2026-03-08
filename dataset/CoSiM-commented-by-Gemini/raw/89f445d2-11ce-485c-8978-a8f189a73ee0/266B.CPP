/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a queue reordering process based on local swaps.
 *
 * @details
 * This program simulates the changes in a queue of boys ('B') and girls ('G')
 * over a specified number of time steps ('t'). In each time step, any boy
 * standing directly in front of a girl swaps places with her. This process is
 * repeated 't' times to find the final arrangement of the queue.
 *
 * @section
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue as a string.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function with an implicit 'int' return type.
 */
main() {
  // n: the number of children in the queue.
  // t: the number of time steps to simulate.
  int n, t;
  cin >> n >> t;
  // str: the initial arrangement of the queue.
  string str;
  cin >> str;

  /**
   * @brief Outer loop that simulates the progression of time for 't' seconds.
   *
   * Invariant: At the beginning of each iteration, 'str' holds the state of the
   * queue at the current time step.
   */
  while (t > 0) {
    /**
     * @brief Inner loop that performs a single pass over the queue to swap adjacent 'BG' pairs.
     *
     * Logic: This sweep identifies all pairs of a boy ('B') immediately followed
     * by a girl ('G') and swaps them.
     */
    for (int i = 0; i < n - 1; i++) {
      // Pre-condition: Check for a 'B' followed by a 'G'.
      if (str[i] == 'B' && str[i + 1] == 'G') {
        swap(str[i], str[i + 1]);
        // cout<<"each "<<str<<endl;
        // Inline: The index 'i' is incremented to prevent the newly-swapped boy
        // (now at i+1) from being considered for another swap in the same time step.
        // This ensures that each child moves at most one position per time step,
        // correctly modeling the simultaneous nature of the swaps.
        i += 1;
      }
    }
    t -= 1;
  }
  // Output the final state of the queue after 't' seconds.
  cout << str << endl;
}
