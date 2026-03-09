/**
 * @file 266A.CPP
 * @brief Counts the number of adjacent identical characters in a string.
 *
 * This program solves a classic introductory problem often framed as "stones on a table".
 * It calculates the minimum number of stones to remove from a row such that no two
 * adjacent stones have the same color. This is equivalent to counting the number of
 * pairs of adjacent, identical characters in a string.
 *
 * @algorithm Single-pass linear scan.
 * @complexity
 * - Time: O(N), where N is the length of the string, because it iterates through the string once.
 * - Space: O(N) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  // n: the number of stones (characters). k: the count of stones to be removed.
  int n, k = 0;
  string s;
  cin >> n >> s;

  /**
   * @brief Block Logic: Iterates through the string to find adjacent identical characters.
   * Invariant: `k` stores the count of adjacent identical characters found so far in the
   *            scanned portion of the string.
   * Note: The loop condition `i < n` relies on the behavior of `std::string` where accessing
   *       `s[n]` yields a null character, which is unlikely to match `s[n-1]`. A more
   *       robust condition would be `i < n - 1`.
   */
  for (int i = 0; i < n; i++) {
    // Pre-condition: Check if the current character and the next one are the same.
    if (s[i] == s[i + 1])
      k++;
  }

  // Output the total number of characters that need to be "removed".
  cout << k << endl;
  return 0;
}