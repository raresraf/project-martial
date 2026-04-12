/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem.
 *
 * This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each second, every instance of a 'B' immediately followed by a 'G' (BG) simultaneously
 * swaps to become 'GB'. The simulation runs for 't' seconds.
 *
 * Algorithm: Discrete-time iterative swap simulation using a character array.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue in a character array.
 */
#include <iostream>

using namespace std;

/**
 * @brief Swaps the values of two integer variables.
 *
 * This is a utility function to exchange the values held by two integer references.
 * Note: While functionally correct, `std::swap` is generally preferred in C++.
 *
 * @param x Reference to the first integer.
 * @param y Reference to the second integer.
 */
void swap(int &x, int &y) {
  int tmp; // Temporary variable to hold one of the values during swap.
  tmp = x; // Store value of x.
  x = y;   // Assign value of y to x.
  y = tmp; // Assign stored value of x to y.
}

/**
 * @brief Main function to simulate the queue reordering.
 *
 * Functional Utility: Reads initial queue configuration and simulation time,
 * then applies swap rules iteratively, and finally outputs the rearranged queue.
 */
int main() {
  int n, t;         // 'n' for queue length (not directly used in loops with queue[j+1]!='\0'), 't' for simulation time.
  char queue[50];   // C-style character array to store the queue string. Max length 49 chars + null terminator.
  cin >> n >> t >> queue; // Read queue length, simulation time, and initial queue string.

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * The outer loop runs for 't' seconds, simulating each second of swaps.
   * Invariant: At the start of each outer loop iteration, `queue` represents the state after `i` seconds.
   */
  for (int i = 0; i < t; i++) {
    int j = 0; // Inner loop counter.
    /**
     * Block Logic: Performs a single-pass sweep to resolve 'BG' to 'GB' swaps within one second.
     * Logic: Iterates through the queue. If a 'B' is found immediately followed by a 'G',
     * they are swapped. The `j++` within the `if` block ensures that the boy who just moved
     * past a girl does not participate in another swap in the same second, adhering to the "simultaneous" rule.
     * Pre-condition: `queue` contains the current state of the queue.
     * Invariant: After each pass, all 'BG' pairs considered for this second have been processed.
     */
    while (queue[j + 1] != '\0') { // Iterate until the second-to-last character.
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        // Swap the 'B' and 'G' characters.
        // Note: The custom 'swap' function is defined for 'int&', but char implicitly converts.
        // It should ideally be 'char&' for type safety, or std::swap should be used.
        swap(queue[j], queue[j + 1]);
        j++; // Increment 'j' to skip the newly swapped 'B'.
        if (queue[j + 1] == '\0') // Check if we reached the end of the string after skipping.
          break;                   // Exit inner loop if at the end.
      }
      j++; // Move to the next character to check.
    }
  }
  cout << queue; // Output the final state of the queue.
  return 0;
}