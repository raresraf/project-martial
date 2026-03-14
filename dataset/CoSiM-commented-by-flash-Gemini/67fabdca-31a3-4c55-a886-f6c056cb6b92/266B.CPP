/**
 * @67fabdca-31a3-4c55-a886-f6c056cb6b92/266B.CPP
 * @brief This program solves Codeforces problem 266B, "Queue at the School".
 * It simulates a queue where boys ('B') and girls ('G') are arranged.
 * For a given number of time steps 't', any boy standing immediately in front
 * of a girl ('BG' pair) will swap places with her. These swaps happen
 * effectively simultaneously within a given second, which this implementation
 * handles by incrementing the inner loop counter `j` by an additional `1`
 * after a swap to skip the newly swapped pair.
 *
 * Algorithm: Iterative simulation. For each second, a single pass through the
 * queue is made. If a 'BG' pair is found, they are swapped, and the scan
 * advances past the swapped pair.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue string.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <string>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to simulate the queue transformation over 't' seconds.
 * Reads the initial queue configuration and applies the swap rules 't' times.
 * @return 0 on successful execution.
 */
int main() {
  int n, t; // Declares 'n' for the length of the queue, and 't' for the number of seconds.
  string s; // Declares a string variable to store the queue configuration.
  
  // Reads 'n' (queue length), 't' (number of seconds), and the initial queue string 's' from standard input.
  cin >> n >> t >> s; 
  
  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: The loop continues 't' times.
   */
  for (int i = 0; i < t; i++) { // Loops 't' times, once for each second.
    /**
     * Block Logic: Inner loop iterates through the queue to find 'BG' pairs and swap them.
     * After a swap, `j` is incremented an additional time to prevent re-evaluating
     * the newly formed 'GB' pair in the same second, effectively simulating simultaneous swaps.
     * Invariant: After this inner loop, all possible 'BG' swaps for the current second are completed.
     */
    for (int j = 0; j < n; ++j) { // Loops through the queue.
      // Pre-condition: Checks if the current element is 'B' and the next is 'G'.
      // Also ensures `j+1` is within string bounds.
      if (s[j] == 'B' && s[j + 1] == 'G') { 
        swap(s[j], s[j + 1]); // Swaps the 'B' and 'G' characters.
        ++j; // Inline: Increments 'j' to skip the character that was just moved from `j+1` to `j`.
             // This prevents a 'B' from moving more than one position in a single second.
      }
    }
  }
  cout << s; // Prints the final state of the queue after 't' seconds.
  return 0; // Indicates successful program termination.
}
