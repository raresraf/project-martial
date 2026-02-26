/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces.
 *
 * @details A "lucky number" contains only the digits '4' and '7'. A number is
 * "nearly lucky" if the count of lucky digits within it is itself a lucky number.
 * This program checks if a given number is nearly lucky.
 *
 * @algorithm
 * 1. Read the input number as a string to allow for easy digit-by-digit access.
 * 2. Iterate through the string, maintaining a counter for the occurrences of
 *    '4' and '7'.
 * 3. After the loop, check if the counter's value is a lucky number. Given the
 *    problem constraints, the only possible lucky counts are 4 and 7.
 * 4. Print "YES" if the count is 4 or 7, and "NO" otherwise. This is done
 *    concisely using a ternary operator.
 *
 * @complexity
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * due to a single pass over the string.
 * Space Complexity: O(L) to store the number as a string.
 */
#include <bits/stdc++.h>

// A collection of personal macros for competitive programming shortcuts.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
#define REP(i, n) FOR(i, 0, n) // A shorthand for a standard for-loop.
#define all(a) a.begin(), a.end()
#define pb push_back

// Type aliases for convenience.
typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

int main() {
  string line;
  getline(cin, line);
  int cnt = 0; // Counter for the number of lucky digits.

  /**
   * @block
   * @brief Count the occurrences of '4' and '7' in the string.
   * The REP macro expands to a standard for-loop.
   */
  REP(i, line.size()) {
    if (line[i] == '4' || line[i] == '7')
      cnt++;
  }

  /**
   * @brief Check if the count is lucky and print the result.
   * @details A ternary operator is used for a compact conditional output.
   * It checks if the count of lucky digits is 4 or 7.
   */
  printf("%s", cnt == 4 || cnt == 7 ? "YES\n" : "NO\n");
  return 0;
}