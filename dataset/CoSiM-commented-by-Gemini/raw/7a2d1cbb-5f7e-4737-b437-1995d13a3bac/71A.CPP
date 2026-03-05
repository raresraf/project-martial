/**
 * @file 71A.CPP
 * @brief Correct solution for Codeforces Problem 71A, "Way Too Long Words".
 * @author Yadi Qian
 * @details This program processes a series of words, shortening any word with
 * a length strictly greater than 10. The file includes a common template of
 * typedefs and headers used in competitive programming, most of which are not
 * utilized for this specific problem.
 *
 * Algorithm:
 * 1. Read the number of words to process, `n`.
 * 2. Loop `n` times. In each iteration, read one word.
 * 3. Check the length of the word.
 * 4. If the length is 10 or less, print the word as is.
 * 5. If the length is greater than 10, print an abbreviation consisting of:
 *    - The first character of the word.
 *    - The number of characters between the first and last (i.e., length - 2).
 *    - The last character of the word.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum word length, due to string I/O.
 * Space Complexity: O(L) to store each word temporarily.
 */
/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  /**
   * @brief Optimizes C++ stream I/O for faster execution in competitive
   * programming contexts.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;
  /**
   * @brief This loop reads and processes each of the `n` words one by one.
   */
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;

    /**
     * @brief Implements the core logic: abbreviate the word if its length is
     * greater than 10, otherwise print it as is.
     */
    if (word.length() <= 10) {
      cout << word << "\n";
    } else {
      cout << word[0] << word.length() - 2 << word[word.length() - 1] << "\n";
    }
  }
  return 0;
}