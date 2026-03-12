/**
 * @5e229cbc-3519-4a76-898b-3a9a36368c4a/266B.CPP
 * @brief This program solves problem 266B, "Queue at the School".
 * It simulates a queue where boys ('B') and girls ('G') are arranged.
 * For a given number of time steps 't', any boy standing immediately in front
 * of a girl ('BG' pair) will swap places with her. These swaps happen
 * simultaneously in a given second.
 *
 * Algorithm: Iterative simulation. For each second, a single pass through the
 * queue is made to perform swaps, advancing the pointer by two positions after
 * a swap to simulate simultaneous movement.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue string.
 */

// Problem: Queue at the school
#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string, algorithm.
using namespace std;     // Uses the standard namespace.

int main(int argc, char const *argv[]) {
  int n, t;    // Declares 'n' for the length of the queue and 't' for the number of seconds.
  cin >> n >> t; // Reads 'n' and 't' from standard input.
  string Queue; // Declares a string variable to store the queue configuration.
  cin >> Queue; // Reads the initial queue configuration from standard input.
  
  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: After each outer loop iteration, the queue reflects the state after one second of swaps.
   */
  for (int i = 0; i < t; i++) { // Loops 't' times, once for each second.
    int j = 0; // Initializes 'j' as an inner loop counter (pointer for queue traversal).
    /**
     * Block Logic: Inner loop iterates through the queue to find 'BG' pairs and swap them.
     * The pointer `j` advances by 2 after a swap to simulate simultaneous movement and
     * prevent re-processing already swapped elements in the same second.
     * Invariant: After this inner loop, all possible 'BG' swaps for the current second are completed.
     */
    while (j < n - 1) { // Loops through the queue. 'j < n - 1' ensures j+1 is valid.
      // Inline: Checks if the current element is 'B' and the next is 'G'.
      if (Queue[j] == (char)'B' && Queue[j + 1] == (char)'G') { 
        swap(Queue[j], Queue[j + 1]); // Swaps the 'B' and 'G' characters.
        j += 2; // Advances 'j' by 2 to skip the swapped pair.
      } else {
        j += 1; // Advances 'j' by 1 if no swap occurs.
      }
    }
  }
  cout << Queue; // Prints the final state of the queue after 't' seconds.
  return 0;      // Indicates successful program termination.
}
