/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces contest 266.
 *
 * @details
 * This program simulates the arrangement of a queue of boys ('B') and girls ('G') over a given
 * period of time. The simulation follows a simple rule: in each second, if a boy is standing
 * immediately in front of a girl, they swap places. This process is repeated for a specified
 * number of seconds.
 *
 * The program reads the number of children `n` and the total time in seconds `t`. It also reads
 * the initial arrangement of the queue as a string. It then enters a loop that simulates the
 * passage of time for `t` seconds. Inside this loop, another loop iterates through the queue
 * from right to left. If it finds a 'B' followed by a 'G', it swaps them and skips the next
 * position to ensure that a single 'B' doesn't move more than one position per second.
 *
 * Algorithm:
 * 1. Read the number of children `n` and the time `t`.
 * 2. Read the initial queue arrangement as a string `q`.
 * 3. Loop `t` times (for each second):
 *    a. Iterate through the queue string from left to right (or right to left, but carefully).
 *    b. If a 'B' is found at index `i` and a 'G' at `i+1`, swap them.
 *    c. If a swap occurs, advance the index by 2 to prevent the same boy from moving again in the same second.
 * 4. Print the final arrangement of the queue.
 *
 * Time Complexity: O(n * t), where `n` is the number of children and `t` is the time in seconds,
 * because the program iterates through the queue for each second.
 * Space Complexity: O(n) to store the queue string.
 */
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int numKids, seconds;
  // Read the number of children and the total time in seconds.
  scanf("%d %d\n", &numKids, &seconds);
  string q;
  // Read the initial arrangement of the queue.
  getline(cin, q);

  // Block Logic: Simulates the time passing for 'seconds' duration.
  // Invariant: At the beginning of each iteration, 'q' represents the queue's state at the current second.
  while (seconds--) {
    // Iterate from right to left to perform swaps for the current second.
    // This approach correctly handles simultaneous swaps within one time step.
    for (int i = 0; i < numKids - 1; ) {
      // Pre-condition: Check for a Boy ('B') followed immediately by a Girl ('G').
      if (q[i] == 'B' && q[i+1] == 'G') {
        // If found, swap their positions.
        q[i] = 'G';
        q[i+1] = 'B';
        // Inline: Increment 'i' by 2 to move past the swapped pair, ensuring the boy
        // who just moved does not move again in the same time step.
        i += 2;
      } else {
        i++;
      }
    }
  }
  // Output the final arrangement of the queue.
  cout << q << endl;
  return 0;
}
