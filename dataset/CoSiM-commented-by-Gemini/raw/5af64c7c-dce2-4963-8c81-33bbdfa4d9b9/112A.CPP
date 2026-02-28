/**
 * @file 112A.CPP
 * @brief Solution for a lexicographical string comparison problem.
 * @details This program reads two strings and compares them lexicographically in a
 * case-insensitive manner. It outputs -1 if the first string is less than the second,
 * 1 if it is greater, and 0 if they are equal.
 *
 * @algorithm Case-insensitive lexicographical comparison.
 * @complexity
 * - Time: O(L), where L is the length of the strings. The program iterates through
 *   the strings at most twice (once for converting to lowercase, once for comparison).
 * - Space: O(L), to store the two input strings.
 */

#include <bits/stdc++.h>

// Macro for long long int type for convenience.
#define ll long long int
// Macro for a common for-loop structure from 0 to k-1.
#define loop(k) for (i = 0; i < k; ++i)
// Macro for a general for-loop structure from k to l-1.
#define loop2(k, l) for (j = k; j < l; ++j)
// A common modulus value used in competitive programming, though not used in this specific problem.
#define mod 1000000007

using namespace std;

int main() {
  /**
   * @brief Optimizes C++ stream I/O.
   * `std::ios_base::sync_with_stdio(false)` disables synchronization with C-style I/O.
   * `cin.tie(NULL)` unties cin from cout.
   */
  std::ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Initialize variables. t is for test cases, but it's a single-run problem.
  ll t = 1, i = 0, j = 0, k = 0;
  // The problem is designed for a single test case, so reading t is commented out.
  // cin>>t;

  // The loop runs once as t is initialized to 1.
  while (t--) {
    string s, s1;
    // Read the two strings to be compared.
    cin >> s >> s1;
    int len = s.length();

    /**
     * @brief Normalize strings to lowercase.
     * This block iterates through both strings and converts each character to
     * lowercase to ensure the comparison is case-insensitive.
     */
    loop(len) {
      s[i] = tolower(s[i]);
      s1[i] = tolower(s1[i]);
    }

    /**
     * @brief Perform lexicographical comparison.
     * This block iterates through the normalized strings character by character.
     * If a difference is found, it prints the result (-1 or 1) and exits the loop.
     */
    loop(len) {
      // Pre-condition: Check if the character in the first string is greater.
      if (s[i] > s1[i]) {
        cout << "1";
        break;
      } 
      // Pre-condition: Check if the character in the first string is smaller.
      else if (s[i] < s1[i]) {
        cout << "-1";
        break;
      }
    }

    /**
     * @brief Check for equality.
     * If the loop completes without breaking, it means all characters were identical.
     * The loop variable 'i' will equal the length of the string.
     */
    if (i == len)
      cout << "0";
    
    cout << "\n";
  }
  return 0;
}
