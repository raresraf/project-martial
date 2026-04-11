/**
 * @file 266A.CPP
 * @brief This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. It uses a `std::vector<char>`
 * to store the stone colors.
 *
 * Algorithm: Greedy approach. Iterate through the stones and count adjacent identical colors.
 * Time Complexity: O(N) where N is the number of stones.
 * Space Complexity: O(N) to store the vector of stones.
 */

#include <bits/stdc++.h> // Includes iostream, vector, etc.

using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of stones in the row.
  int n;
  // Functional Utility: 'ct' (count) accumulates the number of adjacent identical stones that need removal.
  int ct = 0;
  // Functional Utility: 's' is a dynamic array (vector) of characters to store the colors of the stones.
  vector<char> s;

  // Block Logic: Reads the number of stones 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Reads 'n' individual characters (stone colors) and stores them in the vector 's'.
   * Invariant: After each iteration, one character is added to the vector.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: 'c' temporarily stores each character read.
    char c;
    cin >> c;
    s.push_back(c);
  }

  /**
   * Block Logic: Iterates through the vector of stone colors, starting from the second stone,
   * to find adjacent identical colors.
   * Invariant: 'i' represents the current index of the stone being examined. It checks 's[i]'
   *            against 's[i-1]'.
   */
  for (int i = 1; i < n; i++) {
    /**
     * Block Logic: Checks if the current stone and the previous stone have the same color.
     * Pre-condition: 'i' is a valid index, and 'i-1' is also valid.
     * Invariant: If adjacent stones are identical, 'ct' is incremented.
     */
    if (s[i] == s[i - 1])
      ct++;
  }
  // Output the total number of stones that need to be removed.
  cout << ct << endl;
}
