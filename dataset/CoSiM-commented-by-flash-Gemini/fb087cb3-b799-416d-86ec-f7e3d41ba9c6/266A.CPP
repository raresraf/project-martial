/**
 * @file 266A.CPP
 * @brief This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. It uses a C-style string
 * to store the stone colors.
 *
 * Algorithm: Greedy approach. Iterate through the stones and count adjacent identical colors.
 * Time Complexity: O(L) where L is the length of the character array of stones.
 * Space Complexity: O(L) to store the character array of stones.
 */

#include <iostream>
#include <string.h> // Required for C-style string manipulation, though not explicitly used here.

using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of stones in the row.
  int n;
  // Functional Utility: 'count' accumulates the number of adjacent identical stones that need removal.
  int count = 0;
  // Functional Utility: 'str' is a C-style character array to store the colors of the stones.
  char str[50];

  // Block Logic: Reads the number of stones 'n' from standard input.
  cin >> n;
  // Block Logic: Reads the string of stone colors into the character array 'str' from standard input.
  cin >> str;

  /**
   * Block Logic: Iterates through the character array to find adjacent identical colors.
   * Invariant: 'i' represents the current index of the stone being examined. The loop
   *            checks 'str[i]' and 'str[i+1]', so it goes up to n-1.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks if the current stone and the next stone have the same color.
     * Pre-condition: 'i+1' is a valid index within the 'str' array (implicit based on loop limit 'n').
     * Invariant: If adjacent stones are identical, 'count' is incremented.
     */
    if (str[i] == str[i + 1])
      count++;
    // Block Logic: If colors are different, no action is needed, so continue to the next iteration.
    else
      continue;
  }
  // Output the total number of stones that need to be removed.
  cout << count << endl;

  return 0;
}
