/**
 * @file 266B.CPP
 * @brief Simulates a queue reordering process.
 * 
 * This program simulates the reordering of a queue of 'B's (Boys) and 'G's (Girls)
 * over a fixed number of time steps. In each time step, any adjacent 'B' followed
 * by a 'G' will swap places.
 *
 * @details
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the queue.
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
  // Block Logic: Reads the initial state of the queue from standard input.
  for (int i = 0; i < n; i++)
    cin >> a[i];
  if (n < t)
    t = n;
  // Block Logic: Simulates the reordering process for a given number of time steps.
  // Invariant: `a` represents the state of the queue at the beginning of each time step.
  for (int i = 0; i < t; i++) {
    // Block Logic: A single pass through the queue to swap adjacent 'B' and 'G' pairs.
    for (int j = 0; j < n; j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
        // Inline: Skips the next element to prevent a 'G' from moving more than one position per time step.
        j++;
      }
    }
  }
  // Block Logic: Prints the final state of the queue to standard output.
  for (int i = 0; i < n; i++)
    cout << a[i];
  return (0);
}