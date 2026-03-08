/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem on Codeforces.
 *
 * @details
 * This submission solves the problem of abbreviating words that are "too long".
 * A word is considered too long if it has more than 10 characters. The
 * abbreviation consists of the first letter, the number of letters between the
 * first and last, and the last letter. Shorter words are left unchanged.
 *
 * @section
 * Original Submission Metadata (from source file):
 * - SubmissionId: 15988647
 * - ContestId: 71
 * - Index: A
 * - ProblemName: Way Too Long Words
 * - ProblemTags: ['strings']
 * - ProgrammingLanguage: GNU C++
 * - Verdict: OK
 *
 * @section
 * Algorithm: Conditional string manipulation based on length.
 * Time Complexity: O(N), where N is the number of words.
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of words. i, j, k: loop counters and temporary variables.
  // Using 'long long int' is a common competitive programming habit to prevent
  // overflow, though not strictly necessary here. 'j' is unused.
  long long int n, i, j, k;
  cin >> n;
  string a; // 'a' stores the current word.
  /**
   * @brief This loop processes each of the 'n' words.
   */
  for (i = 0; i < n; i++) {
    cin >> a;
    k = a.size();

    /**
     * @brief Abbreviate the word if its length is greater than 10.
     */
    if (k > 10) {
      // Print the abbreviation: first char, middle length, last char.
      cout << a[0] << k - 2 << a[k - 1];
    } else
      // Otherwise, print the word as is.
      cout << a;

    cout << "
";
  }
  return 0;
}
