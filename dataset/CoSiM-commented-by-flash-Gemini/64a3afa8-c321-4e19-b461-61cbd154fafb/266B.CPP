/**
 * @64a3afa8-c321-4e19-b461-61cbd154fafb/266B.CPP
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
 * Space Complexity: O(N) for storing the queue (character array).
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string.h> // Included for C-style string functions, though mainly C++ strings used.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to simulate the queue transformation over 't' seconds.
 * Reads the initial queue configuration and applies the swap rules 't' times.
 * @return 0 on successful execution.
 */
int main() {
  int n, t; // Declares 'n' for the length of the queue and 't' for the number of seconds.

  cin >> n >> t; // Reads 'n' and 't' from standard input.

  char a[n]; // Declares a character array 'a' of size 'n' to store the queue configuration.

  cin >> a; // Reads the initial queue configuration into the array 'a'.

  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: The loop continues 't' times, decrementing 't' in each iteration.
   */
  while (t > 0) {
    /**
     * Block Logic: Inner loop iterates through the queue to find 'BG' pairs and swap them.
     * The pointer `i` advances by 2 after a swap to simulate simultaneous movement and
     * prevent re-processing already swapped elements in the same second.
     * Invariant: After this inner loop, all possible 'BG' swaps for the current second are completed.
     */
    for (int i = 1; i < n; ++i) { // Loop through the queue. 'i' starts at 1 to check 'a[i-1]'.
      // Inline: Checks if the current element is 'G' and the previous is 'B'.
      if (a[i] == 'G' && a[i - 1] == 'B') { 
        char aux;          // Temporary variable for swapping.
        aux = a[i - 1];    // Stores 'B'.
        a[i - 1] = a[i];   // 'B' becomes 'G'.
        a[i] = aux;        // 'G' becomes 'B'.
        i++;               // Advances 'i' by 1 more, effectively skipping the newly swapped pair.
      }
    }
    t--; // Decrements the time counter.
  }

  cout << a << endl; // Prints the final state of the queue, followed by a newline.
}