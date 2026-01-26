/**
 * @file 266A.CPP
 * @brief Solution for a problem that calculates the minimum number of stones
 *        to remove to ensure no two adjacent stones have the same color.
 *
 * The problem involves a row of stones, each with a specific color.
 * The goal is to find how many stones need to be removed so that
 * all remaining adjacent stones have different colors. This is equivalent
 * to counting pairs of adjacent stones with identical colors.
 *
 * Algorithm: Linear scan (iterative comparison of adjacent elements).
 * Time Complexity: O(N) where N is the number of stones.
 * Space Complexity: O(N) for storing the stone colors.
 */

#include <iostream>
#include <string> // Required for string manipulation

using namespace std;

int main() {
  int numOfStones;         // Declares an integer to store the total number of stones.
  int numToRemove = 0;     // Initializes a counter for stones to be removed.
  string stones;           // Declares a string to store the colors of the stones.

  // Reads the total number of stones from standard input.
  cin >> numOfStones;
  // Reads the string representing the colors of the stones (e.g., "RRRGGGBBB").
  cin >> stones;

  /**
   * Block Logic: Iterates through the stones to identify and count adjacent identical colors.
   * Invariant: `numToRemove` stores the count of identical adjacent stone pairs encountered so far.
   */
  for (int i = 1; i < numOfStones; i++) {
    // Compares the current stone with the previous one.
    // If they have the same color, it means one of them needs to be removed.
    if (stones[i - 1] == stones[i]) {
      numToRemove += 1; // Increments the count of stones to remove.
    }
  }

  // Prints the total minimum number of stones that need to be removed.
  cout << numToRemove << endl;
  return 0;
}