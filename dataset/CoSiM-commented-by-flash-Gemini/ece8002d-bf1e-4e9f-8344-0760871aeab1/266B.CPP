/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 *
 * This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each time step, if a boy is directly in front of a girl, they swap positions.
 * This process is repeated for 't' time steps.
 * Algorithm: Iterative simulation. For each time step, a single pass is made through the queue.
 *            When a 'B' is found immediately followed by a 'G', they are swapped, and the
 *            iterator is advanced to prevent double-swapping the same boy in one pass.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue configuration (vector<bool>).
 */

// https://codeforces.com/problemset/problem/266/B
// SOLUTION BY DN54321

#include <algorithm> // Provides std::swap (though manual swap is used here), potentially other utilities.
#include <ctype.h>   // For character manipulation functions (not directly used here but included).
#include <functional> // For function objects (not directly used here but included).
#include <iostream>  // For standard I/O operations (cin, cout, but scanf/printf are used).
#include <string.h>  // For strlen (not directly used here but included).
#include <vector>    // For std::vector.

typedef long long ll; // Alias for long long, not used in this specific solution.
using namespace std;

/**
 * @brief Main function to simulate the queue reordering.
 *
 * Reads initial queue parameters, simulates the boy-girl swaps over 't' time steps,
 * and prints the final queue configuration.
 * @return 0 on successful execution.
 */
int main() {
  int n, t;    // n: length of the queue, t: number of time steps for simulation.
  char g;      // Temporary variable to read character for queue.
  // Block Logic: Read the queue length 'n' and the number of time steps 't'.
  // Pre-condition: User provides two integers.
  // Invariant: 'n' and 't' hold the respective input values.
  scanf("%d%d", &n, &t);

  // Block Logic: Initialize a vector<bool> to represent the queue.
  // 'true' represents a boy ('B'), 'false' represents a girl ('G').
  // Pre-condition: 'n' is valid.
  // Invariant: 'v' is a boolean vector of size 'n', ready to store queue state.
  vector<bool> v(n);

  // Inline: Consume the newline character left in the buffer by the previous scanf.
  getchar(); 

  // Block Logic: Read the initial queue configuration character by character.
  // For each character, populate the boolean vector 'v'.
  // Pre-condition: Input queue string matches 'n' characters.
  // Invariant: 'v' accurately represents the initial queue configuration.
  for (int i = 0; i < n; ++i) {
    scanf("%c", &g);
    if (g == 'B')
      v[i] = 1; // Assign true for Boy
    else
      v[i] = 0; // Assign false for Girl
  }

  // Block Logic: Simulate the queue reordering for 't' time steps.
  // Each outer loop iteration represents one second (time step).
  // Invariant: After each outer loop, the queue state reflects one second of swaps.
  for (int i = 0; i < t; ++i)
    // Block Logic: Iterate through the queue to perform swaps for the current time step.
    // The loop goes up to n-1 to prevent out-of-bounds access when checking v[j+1].
    // Invariant: 'j' iterates through the queue, checking for adjacent 'B' and 'G'.
    for (int j = 0; j < n - 1; ++j) {
      // Block Logic: Check for a boy ('B') immediately followed by a girl ('G').
      // Pre-condition: Current position 'j' and next position 'j+1' are valid.
      if (v[j] && !v[j + 1]) { // If 'v[j]' is true (Boy) and 'v[j+1]' is false (Girl)
        v[j] = 0;             // Boy moves back (becomes Girl's previous spot)
        v[j + 1] = 1;         // Girl moves forward (becomes Boy's previous spot)
        ++j;                  // Inline: Increment 'j' to skip the newly swapped boy,
                              // ensuring he only moves once per second.
      }
    }

  // Block Logic: Print the final configuration of the queue.
  // Iterates through the boolean vector and prints 'B' for true, 'G' for false.
  // Post-condition: The final queue state is printed to standard output.
  for (int i = 0; i < n; ++i) {
    if (v[i])
      printf("B");
    else
      printf("G");
  }
  printf("\n"); // Print a newline at the end of the output.

  return 0; // Indicate successful execution.
}