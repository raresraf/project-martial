/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces (Problem 266B).
 * This program simulates a queue of boys ('B') and girls ('G') where, in each time
 * step, any boy standing in front of a girl ('BG') swaps places with her ('GB').
 * The simulation runs for 't' time steps.
 *
 * Algorithm: Iterative single-pass swap simulation. For each time step, the queue
 * is traversed, and 'BG' pairs are swapped. The inner loop's index is incremented
 * after a swap to prevent a boy from moving multiple positions in one time step.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).
using namespace std;

int main() {
  /**
   * @brief Main function to simulate the queue reordering process.
   * Reads the queue length 'n' and number of time steps 't', then simulates
   * the swapping of 'BG' pairs in the queue for 't' iterations.
   */
  int n, t; // @brief n: number of elements in the queue; t: number of time steps.
  cin >> n >> t;
  string s; // @brief The queue represented as a string of 'B's and 'G's.
  
  // Block Logic: Read the initial queue string character by character.
  // This approach is less common than `cin >> s;` but achieves the same result.
  while (n--) {
    char aux; // @brief Temporary character to read from input.
    cin >> aux;
    s.push_back(aux); // Inline: Append character to the string.
  }
  
  // Block Logic: Orchestrates the temporal progression of the simulation.
  // The outer loop runs for 't' time steps.
  // Invariant: At the start of each iteration, 's' represents the queue state
  //            after the previous time step.
  while (t--) {
    // Block Logic: Performs a single-pass sweep to resolve priority inversions ('BG' pairs).
    // The inner loop iterates through the string, checking adjacent characters.
    for (int i = 0; i < s.size() - 1; i++) {
      // Block Logic: Check for a 'BG' pair that needs swapping.
      if (s[i] == 'B' and s[i + 1] == 'G') {
        s[i] = 'G'; // Inline: Swap 'B' with 'G'.
        s[i + 1] = 'B'; // Inline: Swap 'G' with 'B'.
        // Inline: Skips the next index (i+1) because it has already been processed
        // (the 'B' has moved) and to prevent a single 'B' from migrating
        // more than one position per time step (atomic movement).
        i = i + 1; 
      }
    }
  }
  // Block Logic: Output the final state of the queue after 't' time steps.
  cout << s << endl;
  return 0; // @brief Indicates successful program termination.
}
