/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove to avoid adjacent same-colored stones.
 *
 * This program solves a problem where stones of different colors are arranged
 * in a row. The goal is to find the minimum number of stones to remove so that
 * no two adjacent stones have the same color. This is equivalent to counting
 * the number of pairs of adjacent, identical stones.
 *
 * Algorithm:
 * 1. Read the number of stones, `n`.
 * 2. Read the string `s`, where each character represents the color of a stone.
 * 3. Initialize a counter `c` to 0. This will track the number of stones to remove.
 * 4. Iterate through the string from the first stone to the second-to-last stone.
 * 5. In each step, compare the color of the current stone `s[i]` with the color
 *    of the next stone `s[i+1]`.
 * 6. If they are the same, it means one of these two stones must be removed to
 *    satisfy the condition. The counter `c` is incremented.
 * 7. After the loop completes, `c` holds the total count of removals needed.
 * 8. Print the final count `c`.
 *
 * Time Complexity: O(N), where N is the number of stones, as the program
 * performs a single pass through the string.
 * Space Complexity: O(N) to store the string of stone colors.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  string s;
  int i, n;
  // Read the number of stones and the string representing their colors.
  cin >> n >> s;
  int c = 0; // Counter for stones to be removed.

  /**
   * @brief Loop through the stones to find adjacent pairs of the same color.
   * The loop runs up to n-1 to allow for `s[i+1]` access.
   * Pre-condition: `s` contains the stone colors.
   */
  for (i = 0; i < n - 1; i++) {
    // If the current stone and the next stone have the same color...
    if (s[i] == s[i + 1])
      // ...increment the counter of stones to remove.
      c++;
  }
  
  // Print the total number of removals required.
  cout << c << endl;
}