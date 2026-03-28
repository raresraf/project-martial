/**
 * @file 266B.CPP
 * @brief Simulates a queue rearrangement process.
 *
 * This program simulates a queue of boys ('B') and girls ('G'). Over a given
 * number of time steps (`t`), any boy standing immediately in front of a girl
 * will swap places with her. This process is repeated for the specified duration.
 *
 * @b Algorithm: The simulation uses a nested loop structure. The outer loop
 *    progresses time, and the inner loop sweeps through the queue to perform
 *    swaps. A single pass is made for each time step. When a 'B' followed by
 *    a 'G' is found, they are swapped, and the loop index is incremented to
 *    prevent the newly moved 'G' from swapping again in the same time step.
 *
 * @b Time_Complexity: O(t * n), where 't' is the number of time steps and 'n'
 *    is the length of the queue.
 * @b Space_Complexity: O(n) to store the queue as a string.
 */
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int n, t;
  string queue;
  cin >> n >> t;
  cin >> queue;
  
  // Edge case: if there's only one person, no swaps can occur.
  if (n == 1) {
    cout << queue << endl;
    return 0;
  }
  
  /**
   * @brief Outer loop for time steps.
   * This loop runs for 't' iterations, each representing one unit of time.
   */
  while (t--) {
    /**
     * @brief Inner loop to scan the queue for 'BG' pairs.
     * This loop iterates through the queue to find adjacent 'B' and 'G'
     * and swaps them.
     */
    for (int i = 0; i < queue.length() - 1; i++) {
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        // Swap the boy 'B' and girl 'G'.
        queue[i] = 'G';
        queue[i + 1] = 'B';
        // Increment 'i' to skip the next position, as the 'G' has just moved.
        i++;
      }
    }
  }
  
  cout << queue << endl;
  return 0;
}
