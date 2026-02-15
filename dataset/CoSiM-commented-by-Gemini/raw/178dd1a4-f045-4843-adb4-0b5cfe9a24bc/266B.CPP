/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 * @details This program simulates the swapping of boys ('B') and girls ('G') in a queue.
 * For a given time `t`, any boy standing immediately in front of a girl will swap
 * places with her. This process is repeated for `t` seconds.
 *
 * @section algorithm Algorithm
 * The simulation is performed iteratively:
 * 1. Read the number of children `n`, the total simulation time `t`, and the initial queue string `s`.
 * 2. An outer loop runs `t` times, with each iteration representing one second.
 * 3. An inner loop iterates through the queue to find adjacent 'B' and 'G' pairs (`"BG"`).
 * 4. When a `"BG"` pair is found at indices `i` and `i+1`, the characters are swapped.
 * 5. Crucially, the loop index `i` is then explicitly incremented. This prevents the same boy from moving
 *    more than one position forward in a single time step, ensuring atomic movement.
 * 6. After `t` iterations, the final state of the queue is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(t * n), where `t` is the time and `n` is the length of the queue, due to the nested loops.
 * - Space Complexity: O(n) to store the queue string.
 */
#include "bits/stdc++.h"
// Macro to disable synchronization with C stdio and untie cin/cout for faster I/O in competitive programming.
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
// Macro for redirecting standard I/O to files ("input.txt", "output.txt"), often used for local testing.
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);
// Macro for abstracting the loop for multiple test cases.
#define w(t)                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

using namespace std;

int main(void) {
  // Apply fast I/O settings.
  do_not_sync
  // The FIO macro is commented out, so the program uses standard console I/O.
  // FIO
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  // The outer loop simulates the passage of 't' seconds.
  while (t--) {
    // The inner loop performs a single pass of swaps over the queue for the current second.
    for (int i = 0; i < n - 1; i++) {
      // Check for a 'B' (boy) followed immediately by a 'G' (girl).
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Swap the adjacent 'B' and 'G'.
        swap(s[i], s[i + 1]);
        // Increment 'i' to skip the next position. This is critical to ensure that the boy
        // who just moved does not move again in the same time step (atomic movement).
        i++;
      }
    }
  }

  // Print the final state of the queue.
  cout << s << endl;

  return 0;
}
