/**
 * @file 266B.CPP
 * @brief This program simulates a queue of boys ('B') and girls ('G') where adjacent 'B' and 'G' pairs swap positions over time.
 *   The simulation runs for a given number of seconds, with all possible swaps happening simultaneously in each second.
 *   Problem description: "Queue at the School", Codeforces 266B.
 * * Algorithm: Iterative simulation. For each second, the queue is traversed to identify and execute 'B' and 'G' swaps.
 *   The inner loop's advancement mechanism ensures that a boy who moves back does not immediately swap again in the same second.
 * Time Complexity: O(seconds * kids), where 'seconds' is the number of time steps and 'kids' is the length of the queue.
 *   Given typical constraints (seconds <= 50, kids <= 50), this is very efficient (e.g., 50*50 = 2500 operations).
 * Space Complexity: O(kids) to store the character array representing the queue order.
 */

#include <cstdio>     // Required for printf/scanf, though not used, it's common in CP templates.
#include <iostream>   // Required for std::cin, std::cout.
#include <string>     // Required for std::string, though a char array is used here.
using namespace std;

/**
 * @brief Main function to simulate the queue rearrangement.
 * Functional Utility: Reads the number of kids and seconds, initializes the queue order,
 *                     simulates the swaps over time, and prints the final queue order.
 */
int main() {
  int kids, seconds;
  // Block Logic: Reads the number of kids in the queue and the total simulation time in seconds.
  cin >> kids >> seconds;

  char order[kids]; // Declares a character array to store the initial order of kids in the queue.
  // Block Logic: Reads the initial queue order (e.g., "BGGB").
  cin >> order;

  /**
   * Block Logic: Simulates the passage of time. The outer loop executes for `seconds` iterations,
   *              each iteration representing one second of time.
   * Pre-condition: `seconds` is a non-negative integer.
   * Invariant: After each iteration, the queue reflects the state after one second of swaps.
   */
  for (int i = 0; i < seconds; i++) {

    /**
     * Block Logic: Traverses the queue to identify and perform 'B' and 'G' swaps.
     * Pre-condition: `order` array contains the current queue configuration.
     * Invariant: After each iteration of the inner loop, any `B` followed by a `G` at index `j` has swapped,
     *            and `j` is advanced to correctly handle simultaneous swaps.
     */
    for (int j = 0, k = kids - 1; j < k; j++) {
      /**
       * Block Logic: Checks if a boy ('B') is immediately followed by a girl ('G').
       * If true, they swap positions.
       */
      if (order[j] == 'B' and order[j + 1] == 'G') {
        order[j + 1] = 'B'; // The girl moves to the boy's original position.
        order[j] = 'G';     // The boy moves to the girl's original position.
        // Inline: Increments 'j' to skip the newly moved boy. This prevents a boy
        // from moving multiple positions in a single second, correctly simulating
        // simultaneous swaps for all pairs.
        j++;
      }
    }
  }

  // Block Logic: Prints the final arrangement of the queue after all seconds have passed.
  cout << order;

  return 0;
}
