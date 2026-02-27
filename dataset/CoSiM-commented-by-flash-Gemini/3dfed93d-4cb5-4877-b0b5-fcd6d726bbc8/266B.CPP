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
/*=====TENSHI29=====*/
#include <iostream>
//#include<algorithm>
#include <string>
using namespace std;
int main() {
  // Declare 'n' for the length of the queue, 't' for the number of time units.
  int n, t;
  // Declare string 'q' to represent the queue of 'B' (boys) and 'G' (girls).
  string q;
  // Read the queue length 'n' and the number of time units 't'.
  cin >> n >> t;
  // Read the initial configuration of the queue as a string.
  cin >> q;
  
  // Block Logic: Simulate the queue swapping process for 't' time units.
  // Invariant: After each iteration of this loop, one time unit has passed,
  //            and all possible 'BG' swaps for that unit have occurred.
  while (t--) {
    // Block Logic: Perform a single pass through the queue to identify and execute swaps.
    // Invariant: 'j' iterates through the queue. `j` is incremented twice if a swap occurs
    //            to avoid re-evaluating the newly swapped 'GB' pair in the same time unit.
    for (int j = 0; j < n; j++) {
      // Precondition: Check if a 'B' is immediately followed by a 'G'.
      if (q[j] == 'B' && q[j + 1] == 'G') {
        // Swap 'B' and 'G' characters.
        // The commented out `swap` function is an alternative.
        q[j + 1] = 'B'; // 'B' moves to the right.
        q[j] = 'G';     // 'G' moves to the left.
        j++;            // Increment 'j' again to skip the newly swapped 'B',
                        // preventing it from participating in another swap in this same pass.
      }
    }
  }
  // Block Logic: Print the final configuration of the queue after all time units have passed.
  for (int i = 0; i < n; i++)
    cout << q[i];
  
  return 0; // Indicate successful program termination.
}
