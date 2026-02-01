/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a queue reordering process based on local adjacency rules.
 * This program simulates a line of children ('B' for boy, 'G' for girl) where boys try to move forward
 * in the queue if they are followed by a girl.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the number of time steps and N is the length of the queue.
 * Space Complexity: $O(N)$ to store the character sequence.
 */
// 29.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  int n;
  int t;
  char temp;

  cin >> n >> t;
  char a[n];
  // Block Logic: Read the initial configuration of the queue from input.
  for (int i = 0; i < n; i++)
    cin >> a[i];
  // Conditional Logic: Adjust the number of time steps if it exceeds the queue length.
  // This ensures 't' does not cause out-of-bounds access for array 'a' in the simulation loop.
  if (n < t)
    t = n;
  // Block Logic: Orchestrates the temporal progression of the simulation.
  // The outer loop iterates 't' times, simulating each second of the reordering process.
  // Invariant: At the start of each iteration, the array 'a' represents the queue state at time T-k.
  for (int i = 0; i < t; i++) {
    // Block Logic: Performs a single-pass sweep to resolve priority inversions.
    // This inner loop iterates through the queue, identifying and performing swaps between adjacent 'B' and 'G' characters.
    for (int j = 0; j < n; j++) {
      // Conditional Logic: Checks for an adjacent 'B' (boy) followed by a 'G' (girl).
      // If found, this signifies a reordering condition where the boy moves forward.
      if (a[j] == 'B' && a[j + 1] == 'G') {
        // Block Logic: Perform the swap operation between the boy and the girl.
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
        // Inline: Skips the next index to prevent a single 'B' from migrating more than one position per time step (atomic movement).
        j++;
      }
    }
  }
  // Block Logic: Output the final converged or time-bounded state of the queue.
  for (int i = 0; i < n; i++)
    cout << a[i];
  return (0);
}