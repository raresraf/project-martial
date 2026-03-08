/**
 * @file 112A.CPP
 * @brief Solution for "Petya and Strings" problem (e.g., Codeforces 112-A).
 * @details This program performs a case-insensitive lexicographical comparison of two
 * strings. It uses the standard library functions `std::transform` for case
 * conversion and `std::string::compare` for the comparison itself, which is a
 * very idiomatic C++ approach.
 *
 * @algorithm Case-insensitive lexicographical comparison using `std::string::compare`.
 * @complexity
 * - Time: O(L), where L is the length of the strings. Both `transform` and
 *   `compare` are linear-time operations.
 * - Space: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

// --- Common Competitive Programming Macros & Typedefs ---
typedef long long int ll;
typedef long double ld;
#define mp make_pair
#define v vector
#define pb push_back
#define fi first
#define se second
#define ALL(v) v.begin(), v.end()
#define UN(v)                                                                  
  sort((v).begin(), (v).end()), v.erase(unique(v.begin(), v.end()), v.end())
#define N 31622779
#define mod 998244353

int main() {
  string s1, s2;
  cin >> s1 >> s2;

  /**
   * @block Case Normalization
   * @logic Converts both strings to lowercase to ensure the comparison is case-insensitive.
   */
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  /**
   * @block String Comparison
   * @logic Uses the `std::string::compare` method to lexicographically compare the two strings.
   * This method returns a negative value if the first string is smaller, a positive value
   * if it's larger, and zero if they are equal.
   */
  int x = s1.compare(s2);
  if (x < 0)
    cout << -1 << endl;
  if (x > 0)
    cout << 1 << endl;
  if (x == 0)
    cout << 0 << endl;

  return 0;
}
