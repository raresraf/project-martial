/**
 * @file 281A.CPP
 * @brief Solution for a string capitalization problem.
 * @author Yadi Qian
 *
 * @details This program reads a single word from standard input, capitalizes its
 * first letter, and prints the modified word to standard output.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the input word into a string variable.
 * 2. Access the first character of the string.
 * 3. Convert the character to its uppercase equivalent.
 * 4. Print the modified string.
 *
 * @section complexity
 * Time Complexity: O(L), where L is the length of the input word, dominated by
 * the I/O operations. The capitalization itself is an O(1) operation.
 * Space Complexity: O(L), to store the input word.
 */

#include <bits/stdc++.h>

using namespace std;

// Type aliases and constants for competitive programming.
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
   * Functional Utility: Decouples C++ streams from standard C streams and
   * unties cin from cout to optimize I/O performance in competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string word;
  cin >> word;

  // Block Logic: Capitalize the first letter of the word.
  // Pre-condition: 'word' is not empty.
  // Invariant: Only the first character's case is modified.
  word[0] = toupper(word[0]);

  // Output the resulting word.
  cout << word;

  return 0;
}