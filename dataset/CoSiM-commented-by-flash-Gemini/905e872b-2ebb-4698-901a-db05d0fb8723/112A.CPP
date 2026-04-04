/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison, ignoring case, using `std::string::compare`.
 *
 * This program reads two strings, converts them to lowercase, and then
 * compares them lexicographically using the `std::string::compare` method.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it's larger,
 * and 0 if they are equal.
 *
 * Algorithm: String transformation to lowercase followed by direct lexicographical comparison.
 * Time Complexity: O(L) where L is the length of the longer string, as both `transform`
 *                  and `compare` operations are linear with string length.
 * Space Complexity: O(1) beyond input storage, as `transform` operates in-place.
 */
#include <bits/stdc++.h> // Includes most standard library headers.
#include <iostream>      // Explicitly included for std::cin, std::cout, std::endl.
using namespace std;     // Uses the standard namespace.

typedef long long int ll; // Functional Utility: Defines 'll' as a shorthand for 'long long int'.
typedef long double ld;   // Functional Utility: Defines 'ld' as a shorthand for 'long double'.
#define mp make_pair      // Functional Utility: Defines 'mp' as a shorthand for 'make_pair'.
#define v vector          // Functional Utility: Defines 'v' as a shorthand for 'vector'.
#define pb push_back      // Functional Utility: Defines 'pb' as a shorthand for 'push_back'.
#define fi first          // Functional Utility: Defines 'fi' as a shorthand for 'first' for pairs.
#define se second         // Functional Utility: Defines 'se' as a shorthand for 'second' for pairs.
// Functional Utility: Defines 'ALL(v)' to get iterators for the entire range of a container 'v'.
#define ALL(v) v.begin(), v.end()
// Functional Utility: Defines 'UN(v)' to sort a vector and remove duplicate elements.
#define UN(v)                                                                  \
  sort((v).begin(), (v).end()), v.erase(unique(v.begin(), v.end()), v.end())
// Functional Utility: Defines 'N' as a constant, potentially a maximum size or limit.
#define N 31622779
// Functional Utility: Defines 'mod' as a common modulo value for arithmetic operations.
#define mod 998244353

//  112-A Petya String
int main() {
  /**
   * @brief Main function: Reads two strings, converts them to lowercase, compares them lexicographically,
   *        and prints the comparison result (-1, 1, or 0).
   * @return 0 if the program executes successfully.
   */

  string s1, s2; // Declare two strings for input.
  cin >> s1 >> s2; // Read the two input strings.

  // Functional Utility: Convert all characters in s1 to lowercase in-place.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  // Functional Utility: Convert all characters in s2 to lowercase in-place.
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  // Functional Utility: Lexicographically compares s1 and s2.
  // Returns <0 if s1 is less than s2, >0 if s1 is greater than s2, and 0 if they are equal.
  int x = s1.compare(s2);

  /**
   * @brief Conditional block to print the comparison result.
   * @pre 'x' holds the result of the lexicographical comparison.
   */
  if (x < 0) // If s1 is lexicographically smaller than s2.
    cout << -1 << endl;
  if (x > 0) // If s1 is lexicographically greater than s2.
    cout << 1 << endl;
  if (x == 0) // If s1 and s2 are lexicographically equal.
    cout << 0 << endl;

  return 0;
}