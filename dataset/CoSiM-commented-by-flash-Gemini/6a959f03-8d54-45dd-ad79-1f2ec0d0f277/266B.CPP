/**
 * @6a959f03-8d54-45dd-ad79-1f2ec0d0f277/266B.CPP
 * @brief Problem: Queue at the School / Queue Simulation.
 *        Simulates a queue of boys ('B') and girls ('G').
 *        Over a given number of time steps ('t'), if a boy is immediately
 *        followed by a girl, they swap positions. The task is to determine
 *        the final arrangement of the queue after 't' time steps.
 * Algorithm: Iterative single-pass swap simulation.
 *            For each time step, iterate through the queue. If a 'B' is followed
 *            by a 'G', swap them and advance the index to prevent the same 'B'
 *            from swapping multiple times in a single pass (atomic movement).
 * Time Complexity: O(t * n) where 't' is the number of time steps and 'n' is the queue length.
 *                  The outer loop runs 't' times, and the inner loop runs 'n' times.
 * Space Complexity: O(n) for storing the queue as a character array.
 */
#include <bits/stdc++.h> // Functional Utility: Includes various standard C++ library headers.
using namespace std;

/**
 * @brief Swaps the values of two characters.
 * @param x Pointer to the first character.
 * @param y Pointer to the second character.
 * Pre-condition: 'x' and 'y' point to valid character locations.
 * Post-condition: The characters pointed to by 'x' and 'y' are exchanged.
 */
void swap(char *x, char *y) {
  char t; // Functional Utility: Temporary variable for swapping.
  t = *x;
  *x = *y;
  *y = t;
}

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int i, t, n; // Invariant: 'n' is queue length, 't' is time steps, 'i' is loop index.
  char a[9999]; // Functional Utility: Character array to store the queue. Max size 9998.

  // Block Logic: Reads the queue length 'n' and the number of time steps 't'.
  cin >> n >> t;
  
  // Block Logic: Reads the initial configuration of the queue into character array 'a'.
  // Pre-condition: 'n' is a positive integer.
  // Invariant: After this loop, 'a' contains the initial queue state.
  for (int j = 0; j < n; j++) {
    cin >> a[j];
  }

  // Block Logic: Orchestrates the temporal progression of the simulation.
  // Invariant: At the start of each iteration, the string represents the queue state at time T-l.
  // Pre-condition: 't' is a non-negative integer.
  for (int l = t; l >= 1; l--) {
    // Block Logic: Performs a single-pass sweep to resolve priority inversions within the queue.
    // Invariant: Iterates through the queue, checking for 'B' followed by 'G'.
    // Post-condition: After each inner loop, one time step of swaps has been simulated.
    for (i = 0; i < n; i++) {
      // Block Logic: Checks if a boy ('B') is immediately followed by a girl ('G').
      // Pre-condition: 'i' is a valid index, and 'i+1' is also a valid index within the queue.
      if ((a[i] == 'B') && (a[i + 1] == 'G')) {
        swap(a[i], a[i + 1]); // Functional Utility: Swaps the boy and girl.
        i++; // Inline: Skips the next index to prevent a single 'B' from migrating
             // more than one position per time step (atomic movement).
      }
    }
  }

  // Block Logic: Prints the final state of the queue after all time steps.
  // Pre-condition: The simulation for 't' time steps is complete.
  // Post-condition: The final queue arrangement is outputted to standard output.
  for (int k = 0; k < n; k++) {
    cout << a[k];
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
