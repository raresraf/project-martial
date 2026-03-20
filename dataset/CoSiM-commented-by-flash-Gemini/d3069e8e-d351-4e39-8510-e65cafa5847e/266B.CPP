/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a queue reordering process based on local adjacency rules.
 *        Specifically, it simulates boys ('B') and girls ('G') in a queue where
 *        boys move forward if they are directly behind a girl.
 * Algorithm: Iterative single-pass swap simulation. In each time step, a pass
 *            is made through the queue, swapping adjacent 'BG' pairs to 'GB'.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the
 *                  length of the queue. Each time step involves a linear scan.
 * Space Complexity: O(N) to store the queue string.
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For string manipulation (std::string).

using namespace std;

/**
 * @brief Main function to simulate the queue reordering.
 * Functional Utility: Orchestrates the temporal progression of the queue
 *                     simulation and prints the final state of the queue.
 */
int main() {
  int n;      // Represents the length of the queue.
  int t;      // Represents the number of time steps (iterations) for the simulation.
  char Temp;  // Temporary variable used for swapping characters.
  string Que; // The string representing the queue (e.g., "BGGB").
  
  // Block Logic: Reads the queue length, number of time steps, and the initial queue string.
  cin >> n >> t;
  cin >> Que;
  
  // Block Logic: Orchestrates the temporal progression of the simulation.
  // Invariant: At the start of each iteration, 'Que' represents the queue state after (total_t - t) time steps.
  while (t > 0) {
    // Block Logic: Performs a single-pass sweep to resolve priority inversions
    //              (i.e., 'B' before 'G' will swap).
    // Pre-condition: 'Que' contains 'B' and 'G' characters.
    // Invariant: After each full iteration of this loop, all immediate 'BG' pairs
    //            that could be swapped in this pass have been swapped.
    for (int i = 0; i + 1 < n; i++) {
      if (Que[i] == 'B' && Que[i + 1] == 'G') {
        // Functional Utility: Swaps 'B' and 'G' characters to move 'B' forward.
        Temp = Que[i];
        Que[i] = Que[i + 1];
        Que[i + 1] = Temp;
        // Inline: Increments 'i' to skip the next character, ensuring that a boy
        //         only moves one position per time step (atomic movement) and
        //         prevents double-counting in the same pass.
        i++; 
      }
    }
    --t; // Decrements the time step counter.
  }
  
  // Functional Utility: Prints the final state of the queue after 't' time steps.
  cout << Que;
  return 0; // Indicates successful program termination.
}
