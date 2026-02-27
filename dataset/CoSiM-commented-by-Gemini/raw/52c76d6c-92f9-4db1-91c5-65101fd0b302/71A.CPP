/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details The program processes a list of words. If a word's length is greater
 * than 10 characters, it is abbreviated to its first letter, the count of
 * intermediate characters, and its last letter. Otherwise, it is printed as is.
 *
 * @algorithm
 * 1. Read the number of words, n.
 * 2. Loop n times:
 *    a. Read a word.
 *    b. Check if its length is greater than 10.
 *    c. If it is, print the abbreviated form.
 *    d. If not, print the word unchanged.
 *
 * @complexity
 * Time Complexity: O(n * L), where n is the number of words and L is the
 * maximum length of a word, due to reading the strings. The abbreviation
 * logic itself is O(1) per word.
 * Space Complexity: O(L) to store the longest word.
 */
#include "bits/stdc++.h"

// Collection of common macros for competitive programming.
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define all(v) v.begin(), v.end()
#define allre(v) v.rbegin(), v.rend()
#define sp(x, y) fixed << setprecision(y) << x
#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))
#define GCD(a, b) __gcd(a, b)
#define LCM(a, b) ((a * b) / __gcd(a, b))
using namespace std;

// Note: The constants and the 'Reverse' function are defined but not used in this solution.
const double pi = acos(-1.0);
const double EPS = 1e-6;
const int MOD = (int)1e9 + 7;

bool Reverse(long long a, long long b) { return a > b; }

int main() {
  // IOS macro enables fast I/O.
  IOS 
  
  int n; // Number of words to process.
  cin >> n;
  
  /**
   * @block
   * @brief Loop through each word and apply the abbreviation logic.
   */
  while (n--) {
    string s;
    cin >> s;
    if (s.size() <= 10)
      // If the word is not "too long", print it as is.
      cout << s << endl;
    else
      // Otherwise, print the abbreviated form: first letter, count of
      // intermediate letters, and last letter.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
  }
  return 0;
}
