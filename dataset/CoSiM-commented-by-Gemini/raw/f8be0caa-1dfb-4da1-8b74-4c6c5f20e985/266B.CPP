/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 *
 * Problem Statement:
 * A queue of children is represented by a string of 'B's (boys) and 'G's (girls).
 * In one second, any boy standing immediately in front of a girl swaps places
 * with her. This process is simulated for 't' seconds. The goal is to find the
 * final arrangement of the queue.
 *
 * Algorithm:
 * This program simulates the process step-by-step. It uses a nested loop structure.
 * The outer loop iterates 't' times, representing each second of elapsed time.
 * The inner loop performs a single pass over the queue. During this pass, it
 * identifies every adjacent pair of a boy followed by a girl ('BG') and swaps them.
 *
 * A key part of the logic is that after a swap occurs at index `j` and `j+1`,
 * the loop counter `j` is incremented twice. This ensures that the newly moved
 * girl does not participate in another swap in the same time step, correctly
 * modeling the simultaneous nature of the swaps.
 *
 * Time Complexity: O(t * n), where 't' is the number of seconds and 'n' is the
 * length of the queue.
 * Space Complexity: O(n) to store the queue arrangement.
 */
#include <iostream>

using namespace std;

// Note: This swap function is defined for integers but used for characters.
// While it works due to implicit conversion, a char-specific swap would be more idiomatic.
void swap(int &x, int &y) {
  int tmp;
  tmp = x;
  x = y;
  y = tmp;
}

int main() {
  int n, t;
  char queue[50];
  cin >> n >> t >> queue;

  /**
   * Block Logic: Outer loop to simulate the passage of 't' seconds.
   */
  for (int i = 0; i < t; i++) {
    int j = 0;
    /**
     * Inner Block: A single pass through the queue to perform one second's worth of swaps.
     * The loop continues as long as there is a next character to check against.
     */
    while (queue[j + 1] != '\0') {
      // Find a 'Boy' followed by a 'Girl'.
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        // Perform the swap.
        swap(queue[j], queue[j + 1]);
        // Crucial step: increment j to skip past the two just-swapped characters.
        // This prevents the same girl from moving multiple times in one second.
        j++;
        // This check is redundant as the outer while loop already handles it.
        if (queue[j + 1] == '\0')
          break;
      }
      // Move to the next position in the queue.
      j++;
    }
  }
  
  // Print the final state of the queue.
  cout << queue;
  return 0;
}
