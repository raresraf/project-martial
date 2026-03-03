/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - "Stones on the Table".
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones are of the same color. This is achieved
 * by counting the number of adjacent pairs with identical colors.
 * Algorithm: Single pass iteration.
 * Time Complexity: O(N) - where N is the number of stones (length of the string).
 * Space Complexity: O(N) - to store the input string of stones.
 */
#include <iostream>

using namespace std;

int main() {
  // Declares a string to store the sequence of stones and an integer to count removals.
  string stone;
  int count = 0;
  // Declares an integer to store the total number of stones.
  int n;
  
  // Reads the number of stones (n) and the string representing the stones.
  cin >> n >> stone;
  
  /**
   * Block Logic: Iterates through the stones to identify adjacent pairs of the same color.
   * Invariant: At the start of each iteration, 'count' holds the number of identical
   * adjacent pairs found so far for stones up to index i-1.
   */
  for (int i = 1; i < n; i++) {
    // Compares the current stone with the previous one.
    if (stone[i] == stone[i - 1])
      // If colors are the same, increments the count of stones to be removed.
      count++;
  }

  // Prints the total count of stones that need to be removed.
  cout << count << endl;
  return 0;
}