
/**
 * @file 71A.CPP
 * @author Anmol Gupta
 * @brief Solution to Codeforces problem 71A, "Way Too Long Words".
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first letter, the count of characters
 * between the first and last letters, and the last letter.
 *
 * Algorithm:
 * For each input word, the program checks its length. If the length is greater
 * than 10, it constructs and prints the abbreviation. Otherwise, it prints the
 * word unchanged.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * length of the longest string. Since L is small for each word, it's
 * effectively O(T).
 * Space Complexity: O(L) to store the longest input string.
 */
#include <bits/stdc++.h>
using namespace std;

#pragma GCC optimize("O3") // Request GCC to apply aggressive optimizations.
// Collection of common macros used in competitive programming for brevity.
#define pb push_back
#define debug(x) cout << '>' << #x << " : " << x << endl;
#define mp make_pair
#define ff first
#define ss second
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());
#define endl "\n"
#define MOD 1000000007
#define MOD9 1000000009
#define pin(n) cout << n << endl
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define eps 0.000001
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define bolt ios::sync_with_stdio(0)
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF
#define PI acos(-1.0)
typedef long long ll;

int main() {
  /**
   * Functional Utility: These lines decouple C++ streams from C stdio and
   * untie cin from cout to speed up I/O, which is a common practice in
   * competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  ll t; // Number of test cases.
  cin >> t;

  /**
   * @brief Main processing loop for each test case.
   *
   * Invariant: Executes `t` times, processing one word per iteration.
   */
  while (t--) {
    string str;
    cin >> str;
    int len = str.length();

    /**
     * Block Logic: Implements the abbreviation rule.
     * Pre-condition: `str` contains the input word and `len` is its length.
     * Logic: If the word's length is greater than 10, print the abbreviation
     *        (first char, length-2, last char). Otherwise, print the original word.
     */
    if (len > 10) {
      cout << str[0] << len - 2 << str[len - 1] << endl;
    } else
      cout << str << endl;
  }
}
