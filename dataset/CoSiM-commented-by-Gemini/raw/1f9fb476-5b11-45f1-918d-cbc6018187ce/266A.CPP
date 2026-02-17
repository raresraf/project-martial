/**
 * @file 266A.CPP
 * @brief Solution for the "Stones on the Table" competitive programming problem.
 *
 * @details This program calculates the minimum number of stones to remove from a
 * row so that no two adjacent stones have the same color. The colors are given
 * as a string of characters. The problem is equivalent to counting the number
 * of adjacent pairs of characters that are identical.
 *
 * @section algorithm
 * Algorithm:
 * A greedy approach is used. We iterate through the string of stones from left
 * to right and count every time a stone has the same color as the one
 * immediately following it. This count represents the number of stones that
 * must be removed.
 *
 * 1. Read the number of stones, `n`.
 * 2. Read the string representing the colors of the stones.
 * 3. Initialize a counter for removals to zero.
 * 4. Iterate from the first stone to the second-to-last stone.
 * 5. In each iteration, if the current stone's color is the same as the next
 *    stone's color, increment the counter.
 * 6. Print the final count.
 *
 * @section complexity
 * Time Complexity: O(N), where N is the number of stones (the length of the
 * string), due to the single linear pass through the string.
 * Space Complexity: O(N), to store the input string representing the stones.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  string s;
  cin >> s;

  int count = 0;
  // Block Logic: Iterate through the string to find adjacent identical characters.
  // The loop runs up to n-1 to prevent reading past the end of the string
  // with the s.at(i + 1) access.
  for (int i = 0; i < n - 1; i++)
    // Pre-condition: `i` and `i+1` are valid indices for string `s`.
    // Invariant: `count` stores the number of removals needed for the substring s[0...i].
    if (s.at(i) == s.at(i + 1))
      count++;

  // Output the total number of stones to be removed.
  cout << count << endl;

  return 0;
}
