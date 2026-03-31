/**
 * @file 71A.CPP
 * @brief Provides a concise solution to the "Way Too Long Words" problem.
 *
 * This program abbreviates words that are longer than 10 characters. It reads
 * an integer `t`, the number of test cases. For each case, it reads a word.
 * If the word's length exceeds 10, it is shortened to its first character,
 * the count of its middle characters, and its last character. Otherwise, the
 * word is printed in its original form. This implementation uses modern C++
 * stream features for a compact solution.
 *
 * @author f29155ac-be3b-4b29-bc89-62257d5079b3
 *
 * @algorithm String Size-Checking and Abbreviation
 * @complexity
 * Time: O(T * L), where T is the number of words and L is the max length of a word,
 *       dominated by string input operations.
 * Space: O(L), to store the longest word read.
 */
#include <bits/stdc++.h>
// Common macros for competitive programming
#define fs first
#define sc second
#define pi 3.141592653589793238462643383279502884197
//#define e 2.7182818284590452353602874713526624977572

// Type aliases for convenience
typedef long long int ll;
typedef long double ld;
typedef std::vector<ll> vi;
typedef std::vector<bool> vb;
typedef std::vector<char> vc;
typedef std::pair<ll, ll> pii;
typedef std::vector<pii> vii;

using namespace std;
const int mod = 1e9 + 7; // A common prime modulus

int main() {
  // Optimizes C++ standard streams for faster I/O.
  std::ios::sync_with_stdio(0);
  cin.tie(0);
  // `t` stores the number of words to process.
  int t;
  cin >> t;
  /**
   * @brief Main loop to process each test case (word).
   */
  while (t--) {
    string s;
    cin >> s;
    /**
     * @brief Checks if the word's length is greater than 10.
     * This is the condition for abbreviation.
     */
    if (s.size() > 10)
      // If the word is too long, print its abbreviated form.
      // `std::cout` automatically converts the integer `s.size() - 2` to a string.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << '
';
    else
      // Otherwise, print the word as is.
      cout << s << '
';
  }
}