/*
Hey there stalker....
















Like what you see?
*/
/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem ("Way Too Long Words" - Codeforces 71A).
 * The program reads a number of test cases, and for each test case, it processes a single word.
 * If a word's length exceeds 10 characters, it is abbreviated by taking its first character,
 * the count of intermediate characters, and its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 * Algorithm: String length-based conditional abbreviation.
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 * Space Complexity: O(L) for storing the input string per test case.
 */
#include <algorithm> // Generally useful, but not explicitly used in this specific problem logic.
#include <bits/stdc++.h> // Common competitive programming header, includes many standard libraries.
#include <set>       // Not used in this specific implementation, but often useful.
#include <string>
#include <vector>    // Not used in this specific implementation, but often useful.
using namespace std;

// Macro for fast I/O setup, common in competitive programming.
// Disables synchronization with C stdio and unties cin from cout.
#define IOS                                                                    \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
// Macro for vector push_back, not explicitly used with 'vi' in this code.
#define pb push_back
// Pair of long longs, not used here.
#define pii pair<ll, ll>
// Vector of long longs, not used here.
#define vi vector<ll>
// Returns iterators to the beginning and end of a container, not explicitly used here.
#define all(a) (a).begin(), (a).end()
// First element of a pair, not used here.
#define F first
// Second element of a pair, not used here.
#define S second
// Size of a container, cast to long long, not used here.
#define sz(x) (ll) x.size()
// Loop from a to b-1, not used here.
#define rep(i, a, b) for (ll i = a; i < b; i++)
// Loop from a to b+1 (decreasing), not used here.
#define repr(i, a, b) for (ll i = a; i > b; i--)
// Lower bound, not used here.
#define lbnd lower_bound
// Upper bound, not used here.
#define ubnd upper_bound
// Binary search, not used here.
#define bs binary_search
// Make pair, not used here.
#define mp make_pair
// Long long integer type.
#define ll long long int
// Long double type, not used here.
#define lld long double
// Newline character, optimized for faster output.
#define endl "\n"
// Modulo constant, commonly used in problems involving large numbers.
#define MOD 1000000007

int main() {
  // Functional Utility: Applies fast I/O optimizations defined by the IOS macro.
  IOS

  // Declares 'TESTS' to store the number of test cases.
  int TESTS;
  // Reads the number of test cases.
  cin >> TESTS;
  
  /**
   * Block Logic: Processes each test case.
   * Invariant: At the start of each iteration, 'TESTS' represents the remaining number of test cases to process.
   */
  while (TESTS--) {
    // Declares a string 's' to store the word for the current test case.
    string s;
    // Reads the word.
    cin >> s;
    
    /**
     * Conditional Logic: Determines whether to abbreviate the word or print it as is.
     * Pre-condition: 's' contains a valid string.
     * Post-condition: The abbreviated or original word is printed to standard output.
     */
    if (s.length() > 10) {
      // If the word's length is greater than 10, prints its first character,
      // the number of characters between the first and last (length - 2), and its last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // Otherwise, prints the word without abbreviation.
      cout << s << endl;
    }
  }
  return 0; // Indicates successful program execution.
}
