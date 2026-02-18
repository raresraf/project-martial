// https://codeforces.com/contest/96/problem/A
/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem (96A) from Codeforces.
 * @details The program determines if a situation is "dangerous" by checking for
 * at least 7 consecutive players of the same team ('0' or '1').
 * This solution uses a substring search approach.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution uses a sliding window approach to check for dangerous situations.
 * 1. Define two target strings, "1111111" and "0000000", representing the two
 *    possible dangerous configurations.
 * 2. Read the input string of players.
 * 3. Iterate through the input string with a sliding window of size 7. The loop
 *    runs from the start of the string up to the last possible position where a
 *    7-character substring can start (`n-7`).
 * 4. In each iteration, extract the 7-character substring.
 * 5. Compare the substring with the two target strings.
 * 6. If a match is found, print "YES" and terminate.
 * 7. If the loop finishes without any matches, print "NO".
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string. The loop runs N-6 times,
 * and both `substr` and string comparison take constant time for a fixed length of 7.
 * Space Complexity: O(N) to store the input string.
 */

#pragma GCC optimize("-O3")

#include <bits/stdc++.h>

#include <cmath>

using namespace std;

// An extensive collection of headers, typedefs, and macros typical of a
// competitive programming template.
typedef long long int ll;
typedef long double ld;
typedef pair<int, int> p32;
typedef pair<ll, ll> p64;
typedef pair<double, double> pdd;
typedef vector<ll> v64;
typedef vector<int> v32;
typedef vector<vector<int>> vv32;
typedef vector<vector<ll>> vv64;
typedef vector<p64> vp64;
typedef vector<p32> vp32;
ll MOD = 998244353;
ll NUM = 1e9 + 7;
#define forn(i, e) for (ll i = 0; i < e; i++)
#define forsn(i, s, e) for (ll i = s; i < e; i++)
#define rforn(i, s) for (ll i = s; i >= 0; i--)
#define rforsn(i, s, e) for (ll i = s; i >= e; i--)
#define ln "\n"
#define dbg(x) cout << #x << " = " << x << ln
#define mp make_pair
#define pb push_back
#define ff first
#define ss second
#define INF 1e18
#define fast_cin()                                                             \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL)
#define all(x) (x).begin(), (x).end()
#define sz(x) ((ll)(x).size())
#define zer ll(0)

/**
 * @brief Contains the core logic for solving a single test case.
 */
void solve() {
  string s;
  cin >> s;
  int n = s.length();
  string s1 = "1111111", s2 = "0000000", t;

  /**
   * @block
   * @brief Slides a window of size 7 across the input string to find a match.
   * @pre 's' is the input string of players.
   * @post Prints "YES" and returns if a dangerous situation is found.
   */
  for (int i = 0; i <= n - 7; i++)

  {
    // Extract a substring of length 7.
    t = s.substr(i, 7);
    // Check if the substring consists of 7 identical players.
    if (t == s1 || t == s2)

    {
      cout << "YES" << ln;
      return;
    }
  }

  // If the loop completes, no dangerous situation was found.
  cout << "NO\n";
  return;
}

/**
 * @brief Main function, entry point of the program.
 */
int main()

{
  fast_cin();
  ll t = 1;
  // The code is structured to handle multiple test cases, but it's commented
  // out for this problem.
  // cin>>t;
  while (t--)
    solve();
  return 0;
}