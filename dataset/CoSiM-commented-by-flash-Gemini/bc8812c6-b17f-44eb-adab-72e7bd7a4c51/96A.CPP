/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces (96A).
 * @details The program determines if a situation is "dangerous" by checking if there
 * are at least seven consecutive players of the same team in a line. The teams
 * are represented by '0' and '1'.
 *
 * @section Algorithm
 * The solution uses a sliding window approach to check for consecutive players.
 * 1. It reads the input string representing the line of players.
 * 2. An initial check is performed: if the string length is less than 7, a
 *    dangerous situation is impossible, so it prints "NO" and exits.
 * 3. The program then iterates through the string from the beginning up to the
 *    last position from which a 7-character substring can be formed.
 * 4. In each iteration, it explicitly checks if the 7-character substring starting
 *    at the current position consists of all '0's or all '1's.
 * 5. If such a sequence is found, it immediately prints "YES" and terminates.
 * 6. If the loop completes without finding any dangerous sequence, it prints "NO".
 *
 * @section Complexity
 * - Time Complexity: O(N), where N is the length of the string, as it involves a
 *   single pass with a fixed-size window.
 * - Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
#include <string.h>
using namespace std;
/**
 * @def IOS
 * @brief A common macro in competitive programming to speed up I/O operations by
 * decoupling C++ streams from C standard streams and un-tying cin from cout.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
#define int long long // Redefines 'int' to 'long long' for larger integer range.
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl

/**
 * @brief Main function to read the input and execute the logic.
 */
int32_t main() {
  IOS;
  string str;
  cin >> str;

  /**
   * @brief Pre-condition check: A dangerous situation is impossible if there are
   * fewer than 7 players in total.
   */
  if (str.length() < 7) {
    cout << "NO";
    return 0;
  }

  /**
   * @brief Block Logic: Implements a sliding window of size 7.
   * The loop iterates from the start of the string up to the last possible
   * position (`str.length() - 7`) that can start a 7-character sequence.
   */
  for (int i = 0; i <= str.length() - 7; ++i) {
    /**
     * @brief Logic: Explicitly checks if the substring of length 7 is composed of
     * all '0's or all '1's. If found, the condition is met.
     */
    if ((str[i] == '0') && (str[i + 1] == '0') && (str[i + 2] == '0') &&
        (str[i + 3] == '0') && (str[i + 4] == '0') && (str[i + 5] == '0') &&
        (str[i + 6] == '0')) {
      cout << "YES";
      return 0;
    } else if ((str[i] == '1') && (str[i + 1] == '1') && (str[i + 2] == '1') &&
               (str[i + 3] == '1') && (str[i + 4] == '1') &&
               (str[i + 5] == '1') && (str[i + 6] == '1')) {
      cout << "YES";
      return 0;
    }
  }
  
  // If the loop completes, no dangerous sequence was found.
  cout << "NO";
  return 0;
}
