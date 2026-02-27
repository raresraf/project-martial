/**
 * @file 266B.CPP
 * @brief Solution for competitive programming problem: "Queue at the School".
 *
 * @details This program simulates the process of boys ('B') and girls ('G')
 *          swapping places in a queue. In one unit of time, any boy immediately
 *          followed by a girl swaps places with her. This process is repeated
 *          for a given number of time units.
 *          Algorithm: Iterative single-pass swap simulation.
 *          Time Complexity: O(T * N), where T is the number of time units
 *                           and N is the length of the queue.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n; // Declare 'n' for the length of the queue.
  int t; // Declare 't' for the number of time units.
  // Read the queue length 'n' and the number of time units 't'.
  cin >> n >> t;
  string s; // Declare string 's' to represent the queue of 'B' (boys) and 'G' (girls).
  cin >> s;  // Read the initial configuration of the queue as a string.

  // Block Logic: Simulate the queue swapping process for 't' time units.
  // Invariant: After each iteration of this loop, one time unit has passed,
  //            and all possible 'BG' swaps for that unit have occurred.
  while (t--) {
    // Block Logic: Perform a single pass through the queue to identify and execute swaps.
    // Invariant: 'i' iterates through the queue. If a swap occurs, 'i' is incremented
    //            an additional time to prevent the newly swapped 'B' from participating
    //            in another swap in the same time unit.
    for (int i = 0; i < n - 1; i++) { // Iterate up to n-1 to safely access s[i+1].
      // Precondition: Check if a 'B' is immediately followed by a 'G'.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';     // 'G' moves to the left.
        s[i + 1] = 'B'; // 'B' moves to the right.
        i++;            // Increment 'i' an additional time to skip the newly moved 'B'.
      }
    }
  }
  cout << s; // Output the final configuration of the queue after all time units have passed.
}
