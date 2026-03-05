/**
 * @file 266B.CPP
 * @brief An attempted solution for Codeforces Problem 266B, "Queue at the School".
 * @details The problem requires simulating the state of a queue of boys ('B')
 * and girls ('G') after `t` seconds. In each second, every boy standing
 * immediately in front of a girl swaps places with her.
 *
 * Flawed Algorithm:
 * The code simulates the process over `t` seconds using nested loops. The inner
 * loop iterates through the queue to perform swaps. However, the simulation of
 * the swaps is incorrect. The problem implies that all 'BG' swaps within a
 * single second happen simultaneously. This implementation modifies the queue
 * in-place, which can cause a single 'B' to move multiple positions within
 * what should be a single time step, leading to an incorrect final state.
 *
 * Time Complexity: O(N * T), where N is the number of children and T is the time.
 * Space Complexity: O(N) to store the queue state.
 */
#include <iostream>
using namespace std;

int main() {
  int am, ch;
  cin >> am >> ch;
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  char st[am];
  cin >> st;

  /**
   * @brief Outer loop simulates the passage of time for `ch` seconds.
   */
  for (; ch > 0; ch--) {
    /**
     * @brief Inner loop iterates through the queue to perform swaps for a
     * single time step.
     *
     * @warning The logic here is flawed. It swaps `st[i-1]` and `st[i]`
     * and modifies the loop counter `i` in an attempt to handle swaps
     * correctly, but this does not properly simulate simultaneous swaps.
     * A character can effectively move more than one position in a single
     * time step. For example, in "BGB", the 'B' at index 0 could move to
     * index 2 in one pass of this inner loop, which is incorrect.
     */
    for (int i = 0; i < am; i++) {
      if (st[i] == 'G' && st[i - 1] == 'B' && i != 0) {
        st[i] = 'B';
        st[i - 1] = 'G';
        if (st[i + 1] == 'G')
          i = i + 1;
      }
    }
  }
  cout << st;
  return 0;
}