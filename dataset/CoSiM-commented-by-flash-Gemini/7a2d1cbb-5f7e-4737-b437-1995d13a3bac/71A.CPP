/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, shortening those that are "too long"
 *        according to a specific rule: words with length strictly greater than 10 are replaced
 *        by their first letter, the count of intermediate letters, and their last letter.
 *        Words with length 10 or less are printed without modification.
 * Algorithm:
 *   1. Reads an integer 'n', representing the number of words to process.
 *   2. For each word:
 *      a. Reads the word.
 *      b. If its length is less than or equal to 10, prints the word as is.
 *      c. If its length is greater than 10, prints the first character,
 *         then the numerical value of (length - 2), and finally the last character.
 * Time Complexity: O(L_total), where L_total is the sum of lengths of all input words,
 *                  as each character is processed at most a constant number of times.
 * Space Complexity: O(L_max), where L_max is the maximum length of a single word,
 *                   for storing the current word.
 */
/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0); ///< Constant for Pi.
typedef pair<int, int> PII; ///< Typedef for pair of integers.
typedef vector<int> VI;     ///< Typedef for vector of integers.
typedef vector<string> VS;  ///< Typedef for vector of strings.
typedef vector<PII> VII;    ///< Typedef for vector of pairs of integers.
typedef vector<VI> VVI;     ///< Typedef for vector of vectors of integers.
typedef map<int, int> MPII; ///< Typedef for map with int keys and int values.
typedef set<int> SETI;     ///< Typedef for set of integers.
typedef multiset<int> MSETI; ///< Typedef for multiset of integers.
typedef long int int32;     ///< Typedef for long int.
typedef unsigned long int uint32; ///< Typedef for unsigned long int.
typedef long long int int64; ///< Typedef for long long int.
typedef unsigned long long int uint64; ///< Typedef for unsigned long long int.

int main() {
  /**
   * @brief Main function implementing the solution for the "Way Too Long Words" problem.
   *
   * Sets up fast I/O, then reads a number of words (n). For each word, it applies
   * an abbreviation rule: words strictly longer than 10 characters are replaced by
   * their first letter, the count of intermediate characters, and their last letter.
   * Words with length 10 or less are printed without modification.
   * @return int Returns 0 upon successful execution.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;

    if (word.length() <= 10) {
      cout << word << "\n";
    } else {
      cout << word[0] << word.length() - 2 << word[word.length() - 1] << "\n";
    }
  }
  return 0;
}