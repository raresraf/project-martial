/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid consecutive identical colors.
 * 
 * This program reads an integer representing the number of stones and a string 
 * representing the colors of the stones. It then calculates the minimum number of 
 * stones to remove so that no two adjacent stones have the same color.
 *
 * @b Algorithm: The program iterates through the string of stones and counts the 
 * number of times a stone has the same color as the one preceding it. This count 
 * directly corresponds to the minimum number of stones that need to be removed.
 *
 * @b Time_Complexity: O(N), where N is the number of stones, because it involves a single pass through the string.
 * @b Space_Complexity: O(N) to store the string of stone colors.
 */
#include <iostream>

using namespace std;

int main() {
  int numOfStones;
  int numToRemove = 0;
  string stones;
  // Read the total number of stones.
  cin >> numOfStones;
  // Read the sequence of stone colors.
  cin >> stones;
  /**
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   *
   * Pre-condition: `stones` contains the sequence of colors, `numOfStones` is the length.
   * Invariant: `numToRemove` holds the count of identical adjacent stones found so far in the scanned part of the string.
   */
  for (int i = 1; i < numOfStones; i++) {
    // If a stone has the same color as the previous one, increment the removal counter.
    if (stones[i - 1] == stones[i]) {
      numToRemove += 1;
    }
  }
  // Output the total number of stones to be removed.
  cout << numToRemove << endl;
  return 0;
}
