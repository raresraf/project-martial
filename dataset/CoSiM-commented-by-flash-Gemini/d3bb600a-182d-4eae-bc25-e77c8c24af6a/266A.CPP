/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" competitive programming problem (Codeforces 266A).
 * The problem involves a row of stones, each being one of three colors (Red, Green, Blue).
 * The goal is to find the minimum number of stones to remove so that no two adjacent stones
 * are of the same color. This is achieved by counting pairs of adjacent identical stones.
 * Algorithm: Linear scan to count adjacent identical characters.
 * Time Complexity: O(N), where N is the number of stones (length of the string).
 * Space Complexity: O(N) for storing the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  int n, len_s; // Declares 'n' for the number of stones, 'len_s' is declared but unused.
  int result = 0; // Initializes 'result' to count adjacent identical stones.
  string s;       // Declares 's' to store the string representing the row of stones.
  
  // Reads the number of stones (which is also the length of the string).
  cin >> n;
  // Reads the string of stones.
  cin >> s;
  
  /**
   * Block Logic: Iterates through the string to compare adjacent stones.
   * Invariant: At the start of each iteration, 'i' is the current index (starting from 1),
   * and 'result' holds the count of adjacent identical stones found up to index `i-1`.
   * Pre-condition: 'n' is the length of string 's'.
   * Post-condition: 'result' holds the total count of adjacent identical stones.
   */
  for (int i = 1; i < n; i++) {
    // Conditional Logic: Checks if the current stone is the same color as the previous stone.
    if (s[i - 1] == s[i]) {
      result++; // Increments the count if they are identical.
    }
  }
  // Prints the minimum number of stones to remove (which is 'result').
  cout << result;
  
  return 0; // Indicates successful program execution.
}
