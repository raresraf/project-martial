/**
 * @file 71A.CPP
 * @brief Abbreviates words that are too long.
 *
 * This program reads a number of test cases, and for each case, it reads a
 * word. If the word's length is greater than 10, it's abbreviated by printing
 * its first letter, the number of characters in between, and its last letter.
 * Otherwise, the word is printed as is. This is a common competitive
 * programming problem.
 *
 * @b Algorithm: For each word, its length is checked. If it exceeds 10, an
 *    abbreviation is constructed and printed. Otherwise, the original word is printed.
 * @b Time_Complexity: O(T * L), where T is the number of test cases and L is the
 *    maximum length of a word, due to I/O operations.
 * @b Space_Complexity: O(L) to store the longest input word.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <set>
#include <string>
#include <vector>
using namespace std;
/**
 * @brief Competitive programming macros for brevity.
 * IOS: Speeds up I/O operations.
 * pb: push_back
 * pii: pair<ll, ll>
 * vi: vector<ll>
 * all(a): Returns an iterator to the beginning and end of container 'a'.
 * F, S: first, second for pairs.
 * sz(x): size of container 'x'.
 * rep, repr: for loops.
 * lbnd, ubnd: lower_bound, upper_bound.
 * bs: binary_search.
 * mp: make_pair.
 * ll, lld: long long int, long double.
 * endl: newline character.
 * MOD: A common modulus for modular arithmetic.
 */
#define IOS                                                                    \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define pb push_back
#define pii pair<ll, ll>
#define vi vector<ll>
#define all(a) (a).begin(), (a).end()
#define F first
#define S second
#define sz(x) (ll) x.size()
#define rep(i, a, b) for (ll i = a; i < b; i++)
#define repr(i, a, b) for (ll i = a; i > b; i--)
#define lbnd lower_bound
#define ubnd upper_bound
#define bs binary_search
#define mp make_pair
#define ll long long int
#define lld long double
#define endl "\n"
#define MOD 1000000007
int main() {
  IOS;

  int TESTS;
  cin >> TESTS;
  while (TESTS--) {
    string s;
    cin >> s;
    /**
     * @brief Logic for word abbreviation.
     * If the length is over 10, print the abbreviated form: first letter,
     * the number of middle characters, and the last letter.
     */
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}
