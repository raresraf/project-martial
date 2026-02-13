/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" problem.
 * @details The program calculates the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color. The colors
 * are given as a string of characters.
 *
 * @section algorithm Algorithm
 * A greedy approach is used. We iterate through the stones and count every time
 * two adjacent stones have the same color. This count represents the minimum number of
 * stones that need to be removed to satisfy the condition.
 *
 * 1. Read the number of stones, `n`.
 * 2. Read the string representing the sequence of stone colors.
 * 3. Initialize a counter `count` to 0. This counter will store the number of stones to remove.
 * 4. Iterate through the string from the second character (index 1) to the end.
 * 5. In each step, compare the current stone's color `stone[i]` with the previous one `stone[i-1]`.
 * 6. If they are the same, it means one of them has to be removed, so increment the `count`.
 * 7. After the loop, `count` holds the total minimum number of stones to remove. Print `count`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of stones, due to the single pass through the string.
 * - Space Complexity: O(N), to store the input string of stone colors.
 */
#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string stone;
  int count = 0; // This will count the number of stones to be removed.
  int n;
  cin >> n >> stone; // Read the number of stones and the string representing their colors.

  /**
   * @brief Iterates through the stones, starting from the second one,
   * to compare each stone with its immediate predecessor.
   */
  for (int i = 1; i < n; i++) {
    // If a stone is the same color as the one before it, it's a pair
    // that needs to be broken. We increment the count of stones to remove.
    if (stone[i] == stone[i - 1])
      count++;
  }

  // Output the final count.
  cout << count << endl;
  return 0;
}
