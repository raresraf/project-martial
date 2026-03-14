/**
 * @64db2339-a644-4958-bcdc-04621199ccf7/266B.CPP
 * @brief This program solves problem 266B, "Queue at the School".
 * It simulates a queue where boys ('B') and girls ('G') are arranged.
 * For a given number of time steps 'k', any boy standing immediately in front
 * of a girl ('BG' pair) will swap places with her. These swaps happen
 * simultaneously in a given second.
 *
 * Algorithm: Iterative simulation. For each second, a copy of the queue is made,
 * swaps are identified based on the original state, and applied to the copy.
 * The original queue is then updated with the state from the copy.
 * Time Complexity: O(K * N) where K is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue string and its copy.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to simulate the queue transformation over 'k' seconds.
 * Reads the initial queue configuration and applies the swap rules 'k' times.
 * @return 0 on successful execution.
 */
int main() {
  int n, k; // Declares 'n' for the length of the queue and 'k' for the number of seconds.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.
  string s; // Declares a string variable to store the queue configuration.
  cin >> s; // Reads the initial queue configuration from standard input.
  
  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: The loop continues 'k' times, decrementing 'k' in each iteration.
   */
  while (k--) { // Loops 'k' times, decrementing 'k' in each iteration.
    string z = s; // Inline: Creates a temporary copy 'z' of the current queue state 's'.
                  // This is crucial for simulating simultaneous swaps: all decisions
                  // for the current second are based on the state at the *beginning* of that second ('s'),
                  // and modifications are applied to 'z'.
    /**
     * Block Logic: Inner loop iterates through the queue to find 'BG' pairs and swap them in the copy.
     * Invariant: After this inner loop, all possible 'BG' swaps for the current second are applied to 'z'.
     */
    for (int i = 1; i < s.length(); i++) // Loop through the queue, starting from the second element.
      // Inline: Checks if the previous element is 'B' and the current is 'G' in the *original* state 's'.
      if (s[i - 1] == 'B' && s[i] == 'G') 
        swap(z[i - 1], z[i]); // Swaps the characters in the *copy* 'z'.
    s = z; // Updates the main queue 's' with the state from the processed copy 'z'.
  }
  cout << s << endl; // Prints the final state of the queue, followed by a newline.
  return 0; // Indicates successful program termination.
}
