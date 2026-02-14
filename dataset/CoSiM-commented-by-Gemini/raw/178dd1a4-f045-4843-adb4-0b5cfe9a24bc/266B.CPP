/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 * @details This program simulates the swapping of boys ('B') and girls ('G') in a queue.
 * Over a period of `t` seconds, any boy standing immediately in front of a girl will swap places with her.
 *
 * @section algorithm Algorithm
 * The program performs a direct simulation of the process.
 * 1. It reads the number of children `n`, the simulation time `t`, and the initial arrangement of the queue `s`.
 * 2. An outer `while` loop runs `t` times, with each iteration representing one second.
 * 3. Inside the loop, a `for` loop iterates through the queue to perform swaps.
 *    - It checks for adjacent pairs of 'B' and 'G' (`s[i] == 'B' && s[i+1] == 'G'`).
 *    - When a "BG" pair is found, the `swap` function is called to exchange their positions.
 *    - The loop counter `i` is then incremented. This is a crucial step to ensure that a boy
 *      who has just swapped does not move again in the same time step. This implements the rule
 *      that each swap is an atomic event within the one-second interval.
 * 4. After `t` seconds have passed, the final state of the queue is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(t * n), where `t` is the time and `n` is the length of the queue. This is due to the nested loops.
 * - Space Complexity: O(n) to store the queue string.
 */
#include "bits/stdc++.h"
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);
#define w(t)                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

using namespace std;

/**
 * @brief Main function to execute the queue simulation.
 * @return 0 on successful execution.
 */
int main(void) {
  do_not_sync
      // FIO
      int n,
      t;
  cin >> n >> t;
  string s;
  cin >> s;

  // Simulate the process for 't' seconds.
  while (t--) {
    // Iterate through the queue to perform swaps for the current second.
    for (int i = 0; i < n - 1; i++) {
      // If a boy is in front of a girl, swap them.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        // Increment 'i' to prevent the swapped boy from moving again in the same time step.
        i++;
      }
    }
  }

  // Print the final arrangement of the queue.
  cout << s << endl;

  return 0;
}
