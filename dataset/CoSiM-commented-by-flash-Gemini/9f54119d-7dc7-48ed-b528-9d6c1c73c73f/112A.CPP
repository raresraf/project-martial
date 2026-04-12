/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical string comparison utility.
 * 
 * Logic: Normalizes two input strings to lowercase and performs a character-by-character 
 * comparison to determine their relative dictionary order.
 * 
 * Algorithm: Linear scan with state normalization.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long i;
  string f, s;
  
  /**
   * Functional Utility: Processes pairs of strings from standard input.
   */
  while (cin >> f >> s) {
    long sum1 = 0, sum2 = 0;

    /**
     * Block Logic: State normalization (lowercase conversion).
     * Logic: Iteratively maps uppercase ASCII characters [A-Z] to their 
     * lowercase counterparts [a-z] to ensure case-insensitivity.
     */
    for (i = 0; i < f.size(); i++) {
      if (f[i] >= 'A' && f[i] <= 'Z')
        f[i] = f[i] - 'A' + 97;

      if (s[i] >= 'A' && s[i] <= 'Z')
        s[i] = s[i] - 'A' + 97;
    }

    /**
     * Block Logic: Lexicographical order evaluation.
     * Logic: Performs a sequential comparison of normalized characters. 
     * Returns 1 if string 'f' is greater, -1 if 's' is greater, or 0 if identical.
     */
    for (i = 0; i < s.size(); i++) {
      if (f[i] > s[i]) {
        cout << "1\n";
        return 0;
      }

      if (s[i] > f[i]) {
        cout << "-1\n";
        return 0;
      }
    }
    
    // Output 0 if the loop completes without finding a difference.
    cout << "0\n";
  }
  return 0;
}
