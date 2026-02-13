/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem (Codeforces 266B).
 * @author competitive-programmer
 *
 * @details This program simulates the process of rearranging a queue of boys ('B')
 * and girls ('G'). In each time step, any boy standing immediately in front of a
 * girl swaps places with her. This is repeated for a specified number of time steps.
 *
 * Algorithm:
 * 1. Read the length of the queue (n), the number of time steps (t), and the
 *    initial arrangement of the queue as a string (s).
 * 2. Loop 't' times to simulate each time step.
 * 3. Inside the time step loop, iterate through the queue from left to right.
 * 4. If a 'B' is found immediately before a 'G', swap their positions.
 * 5. After a swap at index `j`, increment `j` an extra time. This is the crucial
 *    step to ensure atomic movement: the boy who just moved to position `j+1`
 *    is not considered for another swap in the same time step.
 * 6. After 't' time steps, print the final arrangement of the queue.
 *
 * Time Complexity: O(t * n), where t is the number of time steps and n is the
 * length of the queue.
 * Space Complexity: O(n) to store the string, or O(1) if input string is not counted.
 */
#include <algorithm>
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  // n: length of the queue, t: number of time steps.
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  // Basic validation to ensure the input string matches the given length.
  if (s.length() != n)
    return -1;

  /**
   * Block Logic: The outer loop simulates the passage of time for 't' seconds.
   * Invariant: At the start of each iteration, 's' holds the queue's state
   * for the current time step.
   */
  for (int i = 0; i != t; ++i) {
    /**
     * Block Logic: The inner loop scans the queue for adjacent 'BG' pairs.
     */
    for (int j = 0; j != n - 1; ++j) {
      // Pre-condition: A 'B' is standing immediately to the left of a 'G'.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        std::swap(s[j], s[j + 1]);
        
        /**
         * Inline Logic: Skip the next position. This ensures that the boy who
         * just moved from `j` to `j+1` is not immediately considered for another
         * swap in the same time step. This makes each boy's movement atomic
         * within a single second.
         */
        if (j + 1 != s.length() - 1) {
          j++;
        }
      }
    }
  }
  
  // Print the final state of the queue.
  cout << s;
  return 0;
}
