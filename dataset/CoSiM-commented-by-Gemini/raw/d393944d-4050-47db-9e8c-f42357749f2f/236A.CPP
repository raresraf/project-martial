/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem (Codeforces 236A) using a sorting-based method.
 * @details This program determines a user's gender based on whether the number of
 * distinct characters in their username is even or odd.
 *
 * @section algorithm
 * Algorithm: Sort and Scan
 * 1. The input string (username) is read.
 * 2. The string is sorted alphabetically using `std::sort`. This operation
 *    groups all identical characters together.
 * 3. The code then makes a single linear pass through the sorted string.
 * 4. It counts the number of times the character changes. This count is
 *    equivalent to the number of distinct characters in the string.
 * 5. The parity of the final count determines the output.
 *
 * @section complexity
 * - Time Complexity: O(L log L), where L is the length of the string. This is
 *   dominated by the `std::sort` operation.
 * - Space Complexity: O(1) auxiliary space, as the sort is done in-place.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  cin >> input;
  int d = 1; // Initialize distinct count to 1 for the first character.

  /**
   * @brief Sorts the string to group identical characters together.
   * This is the core step that enables counting distinct characters via a simple scan.
   */
  sort(input.begin(), input.end());
  
  char t = input[0];
  /**
   * @brief Scans the sorted string to count changes in characters.
   * Each time the character changes, it signifies a new distinct character.
   */
  for (int i = 1; i < input.size(); i++) {
    if (t != input[i]) {
      d++;
      t = input[i];
    }
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * Even: "CHAT WITH HER!", Odd: "IGNORE HIM!".
   */
  if (d % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
