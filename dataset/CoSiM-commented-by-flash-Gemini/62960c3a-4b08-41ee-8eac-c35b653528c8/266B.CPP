/**
 * @62960c3a-4b08-41ee-8eac-c35b653528c8/266B.CPP
 * @brief This program solves problem 266B, "Queue at the School".
 * It simulates a queue where boys ('B') and girls ('G') are arranged.
 * For a given number of time steps 't', any boy standing immediately in front
 * of a girl ('BG' pair) will swap places with her. These swaps happen
 * effectively simultaneously within a given second, meaning a boy can only move
 * once per second.
 *
 * Algorithm: Iterative simulation. A helper function performs a single pass of
 * swaps for one second, and the main function calls it 't' times.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue (char array).
 */
#include <iostream>
using namespace std;

/**
 * @brief Performs a single pass through the queue, swapping 'BG' pairs to 'GB'.
 * This function simulates one second of swaps in the queue.
 *
 * @param a The character array representing the queue.
 * @param n The current length of the queue.
 * @return The first character of the modified queue (the function modifies 'a' in-place).
 *         Note: The return value `*a` is effectively the same as `a[0]` and its usage
 *         in `main` (`*a = cha(a, n)`) is unusual but works due to C++ pointer decay.
 */
char cha(char a[], int n) {
  // Block Logic: Creates a temporary copy of the queue for evaluation.
  // This is crucial for correctly simulating "simultaneous" swaps: all decisions
  // for the current second are based on the state at the *beginning* of that second.
  char b[n]; 
  for (int i = 0; i < n; i++)
    b[i] = a[i];
  
  // Block Logic: Iterates through the copied queue to identify 'BG' pairs.
  // If a 'BG' pair is found, the original queue 'a' is modified.
  for (int i = 0; i < n - 1; i++) {
    if (b[i] == 'B' && b[i + 1] == 'G') { // Pre-condition: Checks 'BG' pair in the *original* state (copy 'b').
      a[i] = 'G';     // Action: Swap 'B' to 'G' in the *actual* queue 'a'.
      a[i + 1] = 'B'; // Action: Swap 'G' to 'B' in the *actual* queue 'a'.
    }
  }
  return *a; // Returns the first element of the modified array.
}

/**
 * @brief Main function to simulate the queue transformation over 't' seconds.
 * Reads the initial queue configuration and applies the swap rules 't' times.
 * @return 0 on successful execution.
 */
int main() {
  int n, t; // Declares 'n' for the length of the queue and 't' for the number of seconds.
  cin >> n >> t; // Reads 'n' and 't' from standard input.
  
  char a[n]; // Declares a character array 'a' of size 'n' to store the queue configuration.
  /**
   * Block Logic: Reads the initial queue configuration into the character array 'a'.
   * Invariant: After this loop, 'a' contains the initial arrangement of boys and girls.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i]; // Reads each character (B or G) into the array.
  
  /**
   * Block Logic: Outer loop simulates the passage of time. The queue state is updated
   * in each iteration to reflect swaps occurring in one second.
   * Invariant: After each outer loop iteration, the queue reflects the state after one second of swaps.
   */
  for (int i = 0; i < t; i++)
    // Inline: Calls the 'cha' function to perform one second's worth of swaps.
    // The assignment `*a = cha(a, n)` is unusual; `cha(a,n)` modifies `a` in-place,
    // and `*a` here only re-assigns `a[0]` to `a[0]`, which has no effect.
    // The intent is solely to call `cha` for its side effect.
    *a = cha(a, n); 
  
  /**
   * Block Logic: Prints the final state of the queue after 't' seconds.
   * Invariant: The characters in array 'a' represent the final arrangement.
   */
  for (int i = 0; i < n; i++)
    cout << a[i]; // Prints each character of the final queue.
}
