/**
 * @file 266A.CPP
 * @brief This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. It processes a string
 * representing the colors of the stones.
 *
 * Algorithm: Iterative scan and comparison with the last non-identical stone.
 * Time Complexity: O(N) where N is the number of stones (length of the string).
 * Space Complexity: O(N) to store the string of stones.
 */

#include <iostream>
#include <string> // Required for std::string
using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of stones in the row.
  int n;
  // Functional Utility: 'pos' tracks the index of the last *unique* stone encountered.
  int pos = 0;
  // Functional Utility: 'cont' (count) accumulates the number of adjacent identical stones that need removal.
  int cont = 0;
  // Functional Utility: 'word' stores the string representing the colors of the stones.
  string word;

  // Block Logic: Reads the number of stones 'n' from standard input.
  cin >> n;
  // Block Logic: Reads the string of stone colors from standard input.
  cin >> word;

  /**
   * Block Logic: Iterates through the string of stone colors, starting from the second stone,
   *              to count adjacent identical colors.
   * Invariant: 'i' represents the current stone being examined. 'pos' points to the last unique stone.
   */
  for (int i = 1; i < n; i++) {
    /**
     * Block Logic: Checks if the current stone has the same color as the stone at 'pos'.
     * Pre-condition: 'i' is a valid index, and 'pos' points to a previous stone.
     * Invariant: If 'word[i]' is identical to 'word[pos]', 'cont' is incremented.
     *            Otherwise, 'pos' is updated to 'i' as a new unique stone is found.
     */
    if (word[i] == word[pos])
      cont++; // Functional Utility: Increments 'cont' if the current stone is identical to the one at 'pos'.
    else
      pos = i; // Functional Utility: Updates 'pos' to the current index 'i' if a new unique stone is found.
  }
  // Output the total number of adjacent identical stones that need to be removed.
  cout << cont << endl;
  return 0;
}
