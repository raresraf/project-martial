/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 *
 * @algorithm
 * This program simulates the arrangement of a queue of boys ('B') and girls ('G')
 * over a period of time `t`. In each second, if a boy is standing directly in
 * front of a girl, they swap places.
 *
 * The simulation proceeds as follows:
 * 1. Read the number of children `n`, the time `t`, and the initial queue
 *    arrangement string.
 * 2. An outer loop runs `t` times, representing each second of time passing.
 * 3. An inner loop iterates through the queue from left to right for each second.
 * 4. Inside the inner loop, it looks for an adjacent pair 'B' followed by 'G'.
 * 5. If such a pair is found at `arr[i]` and `arr[i+1]`, their positions are swapped.
 * 6. After a swap, the inner loop's counter `i` is incremented an extra time.
 *    This is a crucial step to ensure that a boy who has just moved right
 *    (swapped with a girl) is not considered for another swap in the same
 *    time step. This models the swaps as discrete, simultaneous events within
 *    one second.
 * 7. After the outer loop finishes, the final state of the queue is printed.
 *
 * @complexity
 * - Time Complexity: O(N * T), where N is the number of children and T is the time.
 * - Space Complexity: O(N) to store the queue arrangement.
 */
#include <stdio.h>

int main() {
  /* n: the number of children in the queue.
   * t: the total time in seconds for the simulation.
   */
  short n, t;
  scanf("%hd %hd", &n, &t);
  
  /* arr: character array to store the queue arrangement ('B' for boy, 'G' for girl). */
  char arr[n + 1];
  scanf("%s", arr);

  /* Block Logic: Simulate the queue changes over `t` seconds. */
  while (t--) {
    /* Block Logic: A single pass through the queue for one second. */
    for (int i = 0; i < n - 1; i++) {
      // Pre-condition: `arr` is the state at the beginning of the second.
      // Invariant: The sub-array up to `i-1` has been processed for this time step.
      if (arr[i] == 'B' && arr[i + 1] == 'G') {
        // A boy is in front of a girl, so they swap places.
        char aux = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = aux;
        
        // Inline: Skip the next position. After a swap `B G` -> `G B`, the boy is now at `i+1`.
        // We shouldn't consider him for another swap in this same time step.
        // This ensures each child moves at most once per second.
        i++;
      }
    }
  }
  
  // Print the final arrangement of the queue after `t` seconds.
  printf("%s", arr);

  return 0;
}
