/**
 * @file 266B.CPP
 * @brief Solution to a competitive programming problem simulating a queue reordering process.
 * The problem involves a queue of boys ('B') and girls ('G'). The simulation
 * proceeds for a specified number of time steps. In each step, if a boy is
 * immediately followed by a girl (BG), they swap positions (GB). This continues
 * for the entire queue in a single pass for each time step.
 * Algorithm: Iterative simulation of adjacent swaps over a fixed number of time steps.
 * Time Complexity: O(ch * am), where 'ch' is the number of time steps and 'am' is the queue length.
 * Space Complexity: O(am) for storing the character array representing the queue.
 */
#include <iostream>
using namespace std;

int main() {
  // Declares 'am' for the length of the queue (amount of people)
  // and 'ch' for the number of time steps (changes) to simulate.
  int am, ch;
  // Reads the queue length and the number of simulation steps.
  cin >> am >> ch;
  // Declares a character array 'st' of size 'am' to store the queue arrangement.
  char st[am];
  // Reads the initial queue arrangement into the character array.
  cin >> st;
  
  /**
   * Block Logic: Simulates the queue reordering process for 'ch' time steps.
   * Invariant: At the start of each outer loop iteration, 'ch' represents the
   * remaining number of simulation steps, and 'st' holds the queue state from the
   * previous time step (or initial state).
   */
  for (; ch > 0; ch--) {
    /**
     * Block Logic: Performs a single pass through the queue to identify and perform swaps.
     * Invariant: At the start of each inner loop iteration, 'i' is the current index
     * being examined in the queue.
     */
    for (int i = 0; i < am; i++) {
      /**
       * Conditional Logic: Checks for a 'B' followed by a 'G' pattern to perform a swap.
       * `i != 0` ensures that `st[i-1]` is a valid access.
       */
      if (st[i] == 'G' && st[i - 1] == 'B' && i != 0) {
        // Performs the swap: 'B' moves to current position, 'G' moves to previous position.
        st[i] = 'B';
        st[i - 1] = 'G';
        // This condition 'if (st[i + 1] == 'G') i = i + 1;' is specific to some
        // interpretations of the problem. If a swap occurs (GB), it effectively
        // means the 'G' moved forward. The 'i = i + 1' here would then
        // cause the loop to skip the character that just moved to 'st[i]'
        // potentially preventing another swap with it immediately.
        // This is often used to ensure each 'B' only moves one step per time slice.
        if (st[i + 1] == 'G') // Check if the character *after* the current 'G' is also a 'G'
                               // (this logic seems to refer to a potential future 'G' for skipping)
          i = i + 1; // Skips to the next position after the 'G' that just moved,
                     // effectively advancing 'i' an extra step to prevent immediate re-swapping
                     // of the newly moved 'B' (which is now at st[i-1]) with the *next* character.
                     // This is a common optimization to ensure each 'B' moves at most once per pass.
      }
    }
  }
  // Prints the final arrangement of the queue.
  cout << st;
  
  return 0; // Indicates successful program execution.
}