/**
 * @file 266B.CPP
 * @brief Solution to a competitive programming problem: "Queue at the School".
 *
 * Simulates the reordering of a queue containing boys ('B') and girls ('G').
 * In each time step, every boy immediately followed by a girl swaps positions with that girl.
 * This process is repeated for 't' time steps.
 *
 * Algorithm: Iterative string transformation with simultaneous swaps.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue (string).
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to simulate the queue reordering process.
 * Reads the queue length 'n', number of time steps 't', and the initial queue configuration.
 * Simulates 't' steps of 'B' and 'G' swaps and prints the final queue state.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * Declares variables:
   * 'n': The number of elements (boys/girls) in the queue.
   * 't': The number of time steps (iterations) for the queue reordering simulation.
   * 'inp': The string representing the current state of the queue.
   * Reads the values for 'n' and 't' and the initial queue string 'inp' from standard input.
   * Pre-condition: Standard input provides an integer 'n', an integer 't', and a string 'inp' of length 'n'.
   */
  int n, t;
  cin >> n >> t;
  string inp;
  cin >> inp;

  /**
   * Block Logic: Simulates the queue reordering over 't' time steps.
   * Invariant: Each iteration processes one time step of simultaneous swaps.
   * Pre-condition: 'inp' holds the current queue configuration.
   */
  for (int i = 0; i < t; i++) {
    /**
     * Block Logic: Prepares a temporary string for performing simultaneous swaps within a single time step.
     * This ensures all swaps for a given time step are based on the configuration at the *beginning* of that step.
     */
    string temp = inp;
    /**
     * Block Logic: Scans the queue to identify and perform 'B' and 'G' swaps for the current time step.
     * Invariant: 'temp' accumulates the changes for the current time step.
     * Pre-condition: 'temp' is a copy of 'inp' at the start of the time step.
     */
    for (int j = 0; j < n - 1; j++) {
      /**
       * Block Logic: Checks for a 'B' followed by a 'G' to perform a swap.
       * If a 'B' is immediately followed by a 'G', their positions are swapped in the 'temp' string.
       * Pre-condition: 'j' is a valid index within the string bounds for checking adjacent characters.
       */
      if (inp[j] == 'B' && inp[j + 1] == 'G') {
        temp[j] = 'G';
        temp[j + 1] = 'B';
      }
    }
    /**
     * Block Logic: Updates the main queue string with the changes from the current time step.
     * Post-condition: 'inp' reflects the queue state after one time step.
     */
    inp = temp;
  }

  /**
   * Prints the final state of the queue after all time steps have been simulated.
   * Post-condition: The transformed string 'inp' is output to standard output.
   */
  cout << inp;
}
