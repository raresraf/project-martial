/**
 * @file 71A.CPP
 * @brief This program implements the "Way Too Long Words" problem.
 * It abbreviates words longer than 10 characters by replacing the middle characters
 * with their count, keeping the first and last characters. Words 10 characters or shorter are unchanged.
 * Algorithm: String length check and conditional printing/abbreviation using `string.front()`, `string.back()`, `string.size()`.
 * Time Complexity: O(N * L), where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L) for storing the current word.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.
using namespace std;

// Macros for competitive programming convenience.
#define faster ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL) // Optimizes C++ standard streams for faster I/O.
#define mp make_pair                                                         // Shorthand for std::make_pair.
#define mod 1000000007                                                       // Common modulo constant.
#define qmod 998244353                                                       // Another common modulo constant.
#define endl "\n"                                                            // Use newline character instead of std::endl for performance.
#define pb push_back                                                         // Shorthand for std::vector::push_back.
#define ff first                                                             // Shorthand for accessing the first element of a pair.
#define ss second                                                            // Shorthand for accessing the second element of a pair.

// Type definitions for convenience.
typedef long long ll;           // Alias for long long.
typedef pair<int, int> pii;     // Alias for pair of integers.
typedef vector<int> vi;         // Alias for vector of integers.

// Constants definitions.
const int MOD = 1e9 + 7;       // A modulo constant.
const int INF = 1e9 + 5;       // A large integer representing infinity.
const ll LINF = LLONG_MAX;     // A large long long integer representing infinity.

int main() {
  faster; // Applies fast input/output optimizations.
  int n;  // Stores the number of words to process.
  // Reads the number of words that will follow.
  cin >> n;
  /**
   * Block Logic: Processes 'n' words, one in each iteration, applying the abbreviation rule.
   * Words with length greater than 10 are abbreviated; otherwise, they are printed as is.
   * Invariant: Each iteration handles one word according to the abbreviation rules.
   */
  for (int i = 0; i < n; i++) {
    string a; // Stores the current word.
    // Reads the current word into 'a'.
    cin >> a;
    /**
     * Conditional Logic: Checks the length of the current word to decide whether to abbreviate.
     * If the word's length is greater than 10, it's abbreviated.
     * Otherwise, the original word is printed.
     */
    if (a.size() > 10) {
      // Constructs and prints the abbreviated word:
      // first character + (length - 2) as integer + last character.
      cout << a.front() << a.size() - 2 << a.back();
    } else
      // Prints the word as it is, without abbreviation.
      cout << a;

    // Prints a newline character after each word (or its abbreviation).
    cout << "\n";
  }
  // Standard program exit.
  return 0;
}