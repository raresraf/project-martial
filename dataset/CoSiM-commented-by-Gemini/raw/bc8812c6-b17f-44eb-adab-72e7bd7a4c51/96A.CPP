/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 96A "Football".
 * 
 * @details
 * This program determines if a given situation in a football match is "dangerous".
 * A situation is considered dangerous if there are at least seven consecutive players
 * from the same team. The input is a binary string representing the teams of players
 * in a line, where '0' represents one team and '1' represents the other.
 * 
 * @algorithm
 * The core logic employs a single-pass linear scan through the input string. It uses a
 * sliding window of size 7 to check for a contiguous block of seven identical
 * characters ('0' or '1').
 * 
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string. The program
 *   iterates through the string once.
 * - Space Complexity: O(N) to store the input string.
 */

#include <bits/stdc++.h>
#include <string.h>
using namespace std;

/**
 * @def IOS
 * @brief A macro for optimizing C++ standard stream I/O operations.
 * 
 * @details
 * This is a common practice in competitive programming to speed up input and output.
 * `ios::sync_with_stdio(0)` disconnects C++ streams from C standard streams.
 * `cin.tie(0)` and `cout.tie(0)` untie the input and output streams, preventing
 * `cin` from flushing `cout` before each input operation.
 */
#define IOS                                                                    
  ios::sync_with_stdio(0);                                                     
  cin.tie(0);                                                                  
  cout.tie(0);

#define endl "
"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
#define int long long
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    
       << z << endl

/**
 * @brief Main function and entry point of the program.
 */
int32_t main() {
  IOS;
  string str;
  cin >> str;

  /**
   * @brief Pre-condition check.
   * If the total number of players is less than 7, it's impossible to have
   * a sequence of 7, so the situation is not dangerous.
   */
  if (str.length() < 7) {
    cout << "NO";
    return 0;
  }

  /**
   * @brief Sliding window scan.
   * Iterates through the string to check every possible contiguous subsequence of length 7.
   * The loop terminates at `str.length() - 7` to avoid out-of-bounds access.
   */
  for (int i = 0; i <= str.length() - 7; ++i) {
    /**
     * @brief Check for a dangerous sequence.
     * At each position `i`, it checks if the substring of length 7 starting at `i`
     * consists of all '0's or all '1's.
     */
    if ((str[i] == '0') && (str[i + 1] == '0') && (str[i + 2] == '0') &&
        (str[i + 3] == '0') && (str[i + 4] == '0') && (str[i + 5] == '0') &&
        (str[i + 6] == '0')) {
      cout << "YES";
      return 0; // Early exit on finding a match.
    } else if ((str[i] == '1') && (str[i + 1] == '1') && (str[i + 2] == '1') &&
               (str[i + 3] == '1') && (str[i + 4] == '1') &&
               (str[i + 5] == '1') && (str[i + 6] == '1')) {
      cout << "YES";
      return 0; // Early exit on finding a match.
    }
  }

  // If the loop completes without finding a dangerous sequence, the result is "NO".
  cout << "NO";
  return 0;
}
