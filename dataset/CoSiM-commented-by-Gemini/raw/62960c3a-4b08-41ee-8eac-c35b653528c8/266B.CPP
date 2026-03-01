/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 *
 * @problem A queue of 'n' children is represented by a string of 'B' (boys)
 * and 'G' (girls). In one second, any boy standing immediately in front of a
 * girl swaps places with her. This happens for all such 'BG' pairs
 * simultaneously. The task is to find the state of the queue after 't' seconds.
 *
 * @algorithm The problem is solved by simulating the process for 't' discrete time
 * steps.
 * The `cha` function simulates a single time step. To correctly model the
 * simultaneous swaps, it first creates a copy of the queue's state. It then
 * iterates through this copy to identify adjacent 'BG' pairs. When a pair is
 * found in the copy, the swap is performed on the original array. This prevents
 * a single swap from affecting other swap decisions within the same time step.
 * The main function calls this simulation 't' times to get the final result.
 *
 * @time_complexity O(T * N), where T is the number of seconds and N is the number
 * of children in the queue.
 * @space_complexity O(N) to store the queue and a temporary copy for each step.
 */
#include <iostream>
using namespace std;

/**
 * @brief Simulates one second of swaps in the queue.
 * @param a The character array representing the queue (modified in place).
 * @param n The size of the queue.
 * @return The first character of the modified array (this is redundant).
 */
char cha(char a[], int n) {
  // Create a snapshot of the queue at the beginning of the time step.
  char b[n];
  for (int i = 0; i < n; i++)
    b[i] = a[i];

  // Iterate through the snapshot to find 'BG' pairs.
  for (int i = 0; i < n - 1; i++) {
    // Decision is based on the snapshot `b`.
    if (b[i] == 'B' && b[i + 1] == 'G') {
      // The swap is performed on the original array `a`.
      a[i] = 'G';
      a[i + 1] = 'B';
    }
  }
  // Note: This return and the assignment in main() are unnecessary/buggy,
  // as the array `a` is already modified by reference. The code works
  // because the core logic is the in-place modification.
  return *a;
}
int main() {
  int n, t;
  cin >> n >> t;
  char a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // Simulate the process for 't' seconds.
  for (int i = 0; i < t; i++)
    *a = cha(a, n);

  // Print the final state of the queue.
  for (int i = 0; i < n; i++)
    cout << a[i];
}
