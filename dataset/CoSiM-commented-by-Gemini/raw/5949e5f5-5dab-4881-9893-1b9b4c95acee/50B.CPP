/**
 * @file 50B.CPP
 * @brief Solves the "Choosing Symbol Pairs" problem from Codeforces (50B).
 * @details The problem asks to count the number of pairs of indices (i, j) in a string
 * such that the characters at those indices are equal. This is equivalent to
 * calculating the sum of the squares of the frequencies of each unique character
 * in the string. For a character `c` that appears `k` times, it contributes `k*k`
 * to the total count.
 *
 * @algorithm
 * 1. A frequency map (`f`) of size 62 is used to store counts for all possible
 *    alphanumeric characters ('a'-'z', 'A'-'Z', '0'-'9').
 * 2. The code makes a first pass through the input string `s` to populate the
 *    frequency map. It calculates an index for each character to update its count.
 * 3. A second pass is made through the string. For each character `s[i]`, it
 *    looks up its total frequency in the map `f` and adds this frequency to a
 *    running total `ans`.
 * 4. The final `ans` is the sum of `frequency(c)^2` for all unique characters `c`,
 *    which is the solution to the problem.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to two linear
 * passes over the input string.
 * Space Complexity: O(1), as the frequency map has a fixed size (62) regardless
 * of the input string length.
 */
/* .............
   S A K S H A M
   .............
*/
#include <bits/stdc++.h>
using namespace std;
// A large number of template macros and functions follow, not all are used.
#define yes cout << "yes\n";
#define no cout << "no\n";
#define YES cout << "YES\n";
#define NO cout << "NO\n";
#define ll long long int
#define pb push_back
#define mod 1000000007
#define nl cout << "\n";
#define ff first
#define ss second
#define loop1(m) for (int i = 0; i < m; i++)
#define vl vector<ll>
#define vll vector<ll, ll>
#define vc vector<char>
#define msi map<string, int>
#define loop2(n, m) for (auto i = n; i <= m; i++)
#define count1s(n) __builtin_popcount(n)
bool psqrt(ll x) {
  long double sr = sqrt(x);
  return ((sr - floor(sr)) == 0);
}
bool ifodd(ll n) { return (n & 1); }
int lcm(int a, int b) { return a / __gcd(a, b) * b; }
bool isprime(ll n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (int i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  string s;
  cin >> s;
  /**
   * @variable ans: The final count of symbol pairs.
   * @variable f: A frequency array to store the counts of each character.
   * Indices 0-25 are for 'a'-'z', 26-51 for 'A'-'Z', and 52-61 for '0'-'9'.
   */
  ll ans = 0;
  ll f[62] = {0};
  /**
   * @block
   * @brief First pass: Populate the frequency map `f` for all characters in `s`.
   */
  loop1(s.length()) {
    if (s[i] >= 'a' && s[i] <= 'z')
      f[s[i] - 'a']++;
    else if (s[i] >= 'A' && s[i] <= 'Z')
      f[s[i] - 'A' + 26]++;
    else // Character is a digit '0'-'9'.
      f[s[i] - 48 + 52]++;
  }
  /**
   * @block
   * @brief Second pass: Calculate the sum of squares of frequencies.
   * For each character in the string, we add its total frequency to the answer.
   * This results in sum(freq[c]^2 for c in unique_chars).
   */
  loop1(s.length()) {
    if (s[i] >= 'a' && s[i] <= 'z')
      ans += f[s[i] - 'a'];
    else if (s[i] >= 'A' && s[i] <= 'Z')
      ans += f[s[i] - 'A' + 26];
    else // Character is a digit '0'-'9'.
      ans += f[s[i] - 48 + 52];
  }
  cout << ans;
}
