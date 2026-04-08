/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces (Problem 266B).
 * @details This program simulates the state of a queue of boys ('B') and girls ('G')
 * over a given period of time. In each second, any boy standing immediately in front
 * of a girl swaps places with her.
 *
 * @algorithm
 * 1. Read the number of children `n`, the time in seconds `t`, and the initial queue string `s`.
 * 2. Loop `t` times to simulate each second.
 * 3. In each time step, iterate through the queue from the second position to the end.
 * 4. If a 'B' is found at position `i-1` and a 'G' at `i`, swap them.
 * 5. After a swap, increment the loop counter `i` by one extra step. This is crucial
 *    because it ensures that a boy who has just moved does not move again in the
 *    same time step.
 * 6. After `t` seconds, print the final state of the queue string.
 *
 * @time_complexity O(n * t), where `n` is the length of the queue and `t` is the time.
 * @space_complexity O(n) for storing the queue string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n; // Number of children in the queue
  int t; // Time in seconds
  string s; // The string representing the queue
  cin >> n >> t >> s;

  /**
   * @block_logic This outer loop simulates the passage of time. It runs for `t` iterations.
   */
  while (t--) {
    /**
     * @block_logic This inner loop scans the queue for adjacent 'B' and 'G' pairs to swap.
     * @invariant At the end of each inner loop, one second of swaps has been completed.
     */
    for (int i = 1; i < n; ++i) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        // Swap the boy and the girl.
        s[i] = 'B';
        s[i - 1] = 'G';
        // @inline: Increment `i` to skip the next position. This ensures that the
        // boy who just moved cannot move again in the same time step, correctly
        // modeling the rule that each swap takes one second.
        ++i;
      }
    }
  }

  cout << s << endl;
  return 0;
}