/**
 * @5e87a6aa-d4e1-43da-97ae-60ebcca23b41/266B.CPP
 * @brief This program simulates problem 266B, "Queue at the School".
 * It models a queue of boys ('B') and girls ('G'). For a given number of
 * time steps 't', any boy standing immediately in front of a girl ('BG' pair)
 * will swap places with her. This simulation aims to show the queue's state
 * after 't' seconds.
 *
 * Algorithm: Iterative simulation. For each second, a single pass through the
 * queue is made to perform swaps.
 * NOTE: The implementation of advancing the loop pointer `j` by only 1 after
 * a swap might lead to logical inconsistencies if "simultaneous" swaps imply
 * that a boy who just moved forward cannot participate in another swap within
 * the *same second*. A `j += 2` would typically be used for such simultaneous
 * behavior.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string, algorithm.
using namespace std;     // Uses the standard namespace.
typedef long long ll;    // Defines 'll' as a shorthand for long long (unused in this specific solution).

int main() {
  int n, t;    // Declares 'n' for the length of the queue and 't' for the number of seconds.
  scanf("%d%d", &n, &t); // Reads 'n' and 't' from standard input.
  string s;    // Declares a string variable to store the queue configuration.
  cin >> s;    // Reads the initial queue configuration from standard input.

  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: After each outer loop iteration, the queue reflects the state after one second of swaps.
   */
  for (int i = 1; i <= t; i++) { // Loops 't' times, once for each second.
    /**
     * Block Logic: Inner loop iterates through the queue to find 'BG' pairs and swap them.
     * After a swap, the pointer `j` advances by 1.
     * WARNING: Advancing `j` by 1 after a swap (e.g., if 'BG' becomes 'GB') means the 'B'
     * that just moved forward to position `j` might immediately be considered for another
     * swap with `s[j+1]` in the same second, which typically contradicts the notion of
     * simultaneous swaps. For truly simultaneous swaps, `j` should advance by 2.
     * Invariant: All possible 'BG' swaps for the current second are attempted.
     */
    for (int j = 0; j < n - 1; j++) { // Loops through the queue. 'j < n - 1' ensures j+1 is valid.
      // Inline: Checks if the current element is 'B' and the next is 'G'.
      if (s[j] == 'B' && s[j + 1] == 'G') { 
        s[j + 1] = 'B'; // Inline: Swaps the characters by direct assignment.
        s[j] = 'G';     // Original 'B' becomes 'G'.
        j++;            // Advances 'j' by 1. Potential logical inconsistency here.
      }
    }
  }
  cout << s << "\n"; // Prints the final state of the queue after 't' seconds.
  return 0;           // Indicates successful program termination.
}