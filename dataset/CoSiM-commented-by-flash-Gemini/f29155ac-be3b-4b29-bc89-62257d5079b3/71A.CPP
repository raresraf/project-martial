/**
 * @file 71A.CPP
 * @brief This program implements the "Way Too Long Words" problem.
 * It reads 't' words and abbreviates those longer than 10 characters.
 * Abbreviation: First character + (length - 2) as integer + last character.
 * Words 10 characters or shorter are printed unchanged.
 * Algorithm: String length check and conditional printing/abbreviation using string indexing.
 * Time Complexity: O(T * L), where T is the number of test cases and L is the average length of a word.
 * Space Complexity: O(L) for storing the current word.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.

// Macros for competitive programming convenience.
#define fs first           // Shorthand for accessing the first element of a pair.
#define sc second          // Shorthand for accessing the second element of a pair.
#define pi 3.141592653589793238462643383279502884197 // Defines PI, though not used in this specific solution.
// #define e 2.7182818284590452353602874713526624977572 // Defines 'e', commented out in original.

// Type definitions for convenience.
typedef long long int ll;    // Alias for long long integer.
typedef long double ld;      // Alias for long double.
typedef std::vector<ll> vi;  // Alias for vector of long long integers.
typedef std::vector<bool> vb; // Alias for vector of booleans.
typedef std::vector<char> vc; // Alias for vector of characters.
typedef std::pair<ll, ll> pii; // Alias for pair of long long integers.
typedef std::vector<pii> vii;  // Alias for vector of pairs of long long integers.

using namespace std; // Brings all names from the std namespace into the current scope.
const int mod = 1e9 + 7; // A common modulo constant.

int main() {
  // Disables synchronization between C-style I/O and C++ streams, and unties cin from cout.
  // This optimizes input/output operations for competitive programming.
  std::ios::sync_with_stdio(0);
  cin.tie(0);

  int t; // Stores the number of test cases (words to process).
  // Reads the number of test cases.
  cin >> t;
  /**
   * Block Logic: Processes 't' words, one in each iteration, applying the abbreviation rule.
   * Words with length greater than 10 are abbreviated; otherwise, they are printed as is.
   * Invariant: Each iteration handles one word according to the abbreviation rules.
   */
  while (t--) {
    string s; // Stores the current word.
    // Reads the current word into 's'.
    cin >> s;
    /**
     * Conditional Logic: Checks the length of the current word to decide whether to abbreviate.
     * If the word's length is greater than 10, it's abbreviated.
     * Otherwise, the original word is printed.
     */
    if (s.size() > 10)
      // Constructs and prints the abbreviated word:
      // first character (s[0]) + (length - 2) as integer + last character (s[s.size() - 1]).
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << '\n';
    else
      // Prints the word as it is, followed by a newline.
      cout << s << '\n';
  }
  // Standard program exit.
  return 0;
}