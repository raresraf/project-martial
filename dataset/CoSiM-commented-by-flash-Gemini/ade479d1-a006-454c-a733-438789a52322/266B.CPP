/**
 * @file 266B.CPP
 * @brief Solution for Codeforces Problem 266B - "Queue at the School".
 * This program simulates the reordering of a queue of boys ('B') and girls ('G')
 * over a given time 't'. In one second, if a boy is standing immediately in
 * front of a girl, they swap positions. This process is repeated for 't' seconds.
 * Algorithm: Iterative simulation of swaps over time.
 * Time Complexity: O(T * N), where T is the number of seconds and N is the length of the queue.
 * Space Complexity: O(N) - to store the queue string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries
using namespace std;     // Uses the standard namespace.

int main() {
  int n, t;    // n: length of the queue, t: number of seconds.
  cin >> n >> t; // Reads the length of the queue and the time.
  
  string str;  // Declares a string to store the queue arrangement.
  cin >> str;  // Reads the initial queue arrangement.
  
  /**
   * Block Logic: Simulates the swapping process for 't' seconds.
   * The outer loop represents the passage of each second.
   * Invariant: After each outer loop iteration, the queue reflects its state after one second of swaps.
   */
  while (t--) { // Loop 't' times, decrementing 't' in each iteration.
    /**
     * Block Logic: Performs a single pass through the queue to execute all possible swaps in one second.
     * The inner loop iterates from the second element to the end of the queue.
     * Invariant: `str[i-1]` and `str[i]` are considered for a swap. If swapped, `i` is incremented
     *            an additional time to prevent the newly moved boy from swapping again in the same second.
     */
    for (int i = 1; i < n; ++i) { // Loop through the queue from the second element.
      // Checks if a boy ('B') is in front of a girl ('G').
      if (str[i] == 'G' && str[i - 1] == 'B') {
        str[i] = 'B';     // Swap 'G' with 'B'.
        str[i - 1] = 'G'; // Swap 'B' with 'G'.
        ++i;              // Increment 'i' again to skip the newly moved 'B' for this second.
                          // This ensures the boy doesn't immediately swap with the next person in the same second.
      }
    }
  }
  
  // Prints the final arrangement of the queue after 't' seconds.
  cout << str;
  
  return 0; // Indicates successful program execution.
}
