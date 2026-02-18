/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (266B) from Codeforces.
 * @details This program simulates a queue of boys ('B') and girls ('G'). Over a given
 * number of time steps 't', any boy standing directly in front of a girl swaps
 * places with her. This version uses C-style arrays and I/O.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution is a discrete-time simulation.
 * 1. Read the length of the queue (n), the number of time steps (t), and the initial queue string (s).
 * 2. An outer loop runs for `t` iterations, representing each second of time.
 * 3. An inner loop scans the queue from left to right to identify all 'BG' pairs.
 * 4. When a 'BG' pair is found at indices `i` and `i+1`, their positions are swapped.
 * 5. To ensure that each child moves at most once per time step, the loop index `i` is
 *    incremented inside the `if` block. This, combined with the `for` loop's own increment,
 *    skips the next position, preventing the swapped boy from moving again in the same time step.
 * 6. After `t` seconds, the final state of the queue is printed.
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * The outer loop runs T times, and the inner loop scans the array of length N.
 * Space Complexity: O(N) to store the queue as a character array.
 */
#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n, t;

  scanf("%d %d", &n, &t);

  // A variable-length array (VLA) is used here. This is a C99 feature
  // and a non-standard extension in C++.
  char s[n];

  scanf("%s", s);

  /**
   * @block
   * @brief Simulates the swapping process over 't' seconds.
   * @pre 't' is the number of time steps, 's' is the initial queue.
   * @post 's' represents the final queue arrangement after 't' seconds.
   */
  while (t--) {
    /**
     * @block
     * @brief A single pass through the queue for one time step.
     * @invariant The portion of the array s[0...i-1] is in its state for the end of this time step.
     * @pre 's' is the state of the queue at the beginning of the current time step.
     */
    for (int i = 0; i < (n - 1); i++) {
      // If a Boy is in front of a Girl, they swap.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        char c = s[i];
        s[i] = s[i + 1];
        s[i + 1] = c;

        // Skip the next position, as the swapped Boy cannot move again in this time step.
        i++;
      }
    }
  }

  printf("%s\n", s);

  return 0;
}
