/**
 * @file raw/4561cb55-f3d7-4dc1-9161-9e4d7cb1f1d2/266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 *
 * @algorithm
 * The problem describes a discrete-time simulation of a queue of boys ('B')
 * and girls ('G'). In each time step (second), any boy standing immediately
 * in front of a girl swaps places with her. These swaps happen simultaneously.
 *
 * This program simulates this process for a given number of seconds `t`.
 *
 * The algorithm is as follows:
 * 1. Read `n` (queue length) and `t` (number of seconds).
 * 2. Read the initial queue arrangement into a character array.
 * 3. Start an outer loop that iterates `t` times, representing each second.
 * 4. Inside, an inner loop iterates through the queue from left to right to
 *    find all 'BG' pairs.
 * 5. If a 'B' is found at index `i` and a 'G' at `i+1`, they are swapped.
 * 6. After a swap, the inner loop's counter `i` is incremented an extra time.
 *    This is a critical step. It ensures that after the 'B' moves to position
 *    `i+1`, it is not considered for another swap in the same time step. This
 *    correctly models the simultaneous nature of the swaps within a single
 *    second.
 * 7. After the outer loop completes, the final arrangement of the queue is printed.
 *
 * @complexity
 * - Time Complexity: O(t * n), where `t` is the number of seconds and `n` is
 *   the length of the queue. This is due to the nested loops.
 * - Space Complexity: O(n), to store the character array for the queue.
 */
#include <stdio.h>

int main() {
  short n, t; // n: queue length, t: time in seconds.
  scanf("%hd %hd", &n, &t);
  
  // arr: character array to store the queue arrangement.
  // n+1 size to accommodate the null terminator for string operations.
  char arr[n + 1];
  scanf("%s", arr);

  /**
   * @brief Outer loop to simulate the passage of time for 't' seconds.
   */
  while (t--) {
    /**
     * @brief Inner loop to scan the queue and perform swaps for the current second.
     */
    for (int i = 0; i < n - 1; i++) {
      // Check for a boy ('B') immediately followed by a girl ('G').
      if (arr[i] == 'B' && arr[i + 1] == 'G') {
        // Swap their positions.
        char aux = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = aux;
        
        // Crucial step: increment 'i' to move past the swapped pair.
        // This prevents the newly moved 'B' from swapping again in the same time step,
        // correctly modeling simultaneous movement.
        i++;
      }
    }
  }

  // Print the final state of the queue.
  printf("%s", arr);

  return 0;
}
