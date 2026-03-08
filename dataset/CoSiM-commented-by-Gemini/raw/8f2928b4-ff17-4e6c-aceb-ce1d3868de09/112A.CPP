/**
 * @file 112A.CPP
 * @brief Solution for case-insensitive lexicographical string comparison (e.g., Codeforces 112A).
 * @details This program compares two strings, ignoring case, and outputs -1, 1, or 0
 * based on whether the first string is lexicographically smaller than, larger than,
 * or equal to the second. It uses a clever arithmetic trick to determine the result.
 *
 * @algorithm Case-insensitive lexicographical comparison.
 * @complexity
 * - Time: O(L), where L is the length of the strings. This is dominated by the
 *   `transform` operation and the comparison loop.
 * - Space: O(L) to store the two input strings.
 */
#include <algorithm>
#include <iostream>

using namespace std;

// NOTE: This macro is syntactically incorrect and would cause a compile error.
// It should be "#define ll long long". It is unused in this program.
#define long long ll;

// Global variables are common in competitive programming but are generally discouraged.
string s1, s2;

/**
 * @brief Performs the core comparison logic.
 */
void solve() {
  int i = 0; // Index for iterating through the strings.
  int j = 0; // Stores the result: -1, 0, or 1.
  bool found = false;

  /**
   * @block Character-by-Character Comparison
   * @logic The loop proceeds as long as the strings have characters to compare and
   * no difference has been found yet.
   */
  while (i < s1.length() && i < s2.length() && !found) {
    int c1 = int(s1[i]);
    int c2 = int(s2[i]);
    if (c1 != c2) {
      /**
       * @brief Signum Function Trick
       * @logic This arithmetic expression calculates the sign of the difference.
       * - If c1 > c2, (positive / positive) = 1.
       * - If c1 < c2, (negative / positive) = -1.
       * This is a compact but less readable alternative to an if-else block.
       */
      j = (c1 - c2) / abs(c1 - c2);
      found = true; // Stop after the first difference is found.
    }
    i++;
  }
  cout << j;
}

int main() {
  cin >> s1 >> s2;

  /**
   * @block Case Normalization
   * @logic Uses `std::transform` with a lambda function to convert both strings
   * to lowercase, ensuring a case-insensitive comparison.
   */
  transform(s1.begin(), s1.end(), s1.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });
  transform(s2.begin(), s2.end(), s2.begin(),
            [](unsigned char c) -> unsigned char { return tolower(c); });

  solve();
}
