/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A, "Stones on the Table".
 * @details The program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. This is achieved by
 * counting every pair of adjacent stones that are of the same color.
 *
 * @algorithm Single-pass linear scan. The algorithm iterates through the sequence
 * of stones once, comparing each stone with the one preceding it.
 * @time_complexity O(n), where 'n' is the number of stones, as the loop runs n-1 times.
 * @space_complexity O(n) to store the input string representing the stone colors.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  int n; // The total number of stones.
  string word; // A string representing the colors of the stones.
  int to_take = 0; // Counter for the minimum number of stones to remove.
  
  cin >> n >> word;
  
  /**
   * Block Logic: Iterate through the stones starting from the second stone
   * to compare each stone with its immediate predecessor.
   */
  for (int i = 1; i < n; i++) {
    // Pre-condition: Check if the current stone has the same color as the previous one.
    if (word[i] == word[i - 1]) {
      // If they are the same, one must be removed. Increment the counter.
      to_take++;
    }
  }

  // Output the total number of stones that need to be taken.
  cout << to_take << endl;

  return 0;
}