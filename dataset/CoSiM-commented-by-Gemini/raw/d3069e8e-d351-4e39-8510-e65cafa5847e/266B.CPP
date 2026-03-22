/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem (Codeforces 266B).
 * @details This program simulates a discrete-time process of reordering a queue.
 * The queue consists of boys ('B') and girls ('G'). At each time step, if a
 * boy is standing immediately in front of a girl, they swap places. This is
 * repeated for a given number of seconds.
 *
 * @section algorithm
 * Algorithm: Iterative single-pass swap simulation.
 * The simulation proceeds for T time steps. In each step, a single pass is
 * made through the queue. During the pass, any adjacent 'BG' pair is swapped.
 * A crucial detail is that after a swap, the loop index is advanced an extra
 * step to ensure a single person moves at most once per time step.
 *
 * @section complexity
 * - Time Complexity: O(T * N), where T is the number of time steps and N is
 *   the length of the queue.
 * - Space Complexity: O(N) to store the queue as a string.
 */
#include <iostream>
using namespace std;
int main() {
  int n, t;
  char Temp;
  string Que;
  cin >> n >> t;
  cin >> Que;

  /**
   * @brief This outer loop simulates the passage of time for 't' seconds.
   * @invariant At the start of each iteration, `Que` holds the state of the
   * queue for the current time step.
   */
  while (t > 0) {
    /**
     * @brief This inner loop performs a single pass over the queue.
     * It identifies and executes all possible adjacent 'BG' swaps for the
     * current second.
     */
    for (int i = 0; i + 1 < n; i++) {
      // Pre-condition: Check for a Boy ('B') followed immediately by a Girl ('G').
      if (Que[i] == 'B' && Que[i + 1] == 'G') {
        // Block Logic: Perform the swap.
        Temp = Que[i];
        Que[i] = Que[i + 1];
        Que[i + 1] = Temp;
        
        // Inline: Skips the next index. This is critical to ensure that the boy
        // who just moved does not immediately participate in another swap in
        // the same time step. It enforces atomic movement of one position
        // per second per person.
        i++;
      }
    }
    --t;
  }
  
  // Output the final state of the queue after T seconds.
  cout << Que;
  return 0;
}
