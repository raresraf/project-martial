/**
 * @file 118A.CPP
 * @brief Determines if two strings can be made equal by swapping exactly one pair of characters.
 * @details The filename `118A.CPP` is likely a misnomer, as the logic does not correspond to
 * Codeforces Problem 118A ("String Task"). Instead, it solves the "single swap" string problem.
 *
 * Algorithm:
 * 1. Read two strings, `s1` and `s2`.
 * 2. If the strings are not of the same length, they cannot be made equal by a swap.
 * 3. Find all indices where the characters of `s1` and `s2` differ.
 * 4. If the number of differing indices is not exactly 2, it's impossible with a single swap.
 *    - 0 differences mean the strings are already equal.
 *    - 1 difference is impossible to resolve with one swap.
 *    - More than 2 differences require more than one swap.
 * 5. If there are exactly two differing indices, check if swapping the characters at these
 *    positions in `s1` makes it identical to `s2`.
 * 6. If all conditions for a single swap are met, output "YES"; otherwise, output "NO".
 *
 * Time Complexity: O(N), where N is the length of the strings.
 * Space Complexity: O(1) (or O(k) where k is the number of differences, but k is bounded by 2).
 */
#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("-O3")

#define ll long long int
#define pll pair<ll, ll>
#define pls pair<ll, string>
#define psl pair<string, ll>
#define plc pair<ll, char>
#define pcl pair<char, ll>
#define pss pair<string, string>
#define maxheap(type) priority_queue<type>
#define minheap(type) priority_queue<type, vector<type>, greater<type>>
#define setbits(x) __builtin_popcountll(x)
#define zerobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define precise(x, y) fixed << setprecision(y) << x
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define rep(n) for (auto i = 0; i < n; i++)

int main() {
  /**
   * @brief Optimizes C++ stream I/O for speed.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  string s1, s2;
  cin >> s1 >> s2;

  /**
   * @brief Pre-condition: Strings must be of equal length for a single swap to work.
   */
  if (s1.size() != s2.size()) {
    cout << "NO\n";
    return 0;
  }
  ll n = s1.size();
  vector<ll> st;
  /**
   * @brief This loop identifies all positions where the two strings differ.
   * It stores the indices of mismatches in the `st` vector.
   * The loop exits early if more than two mismatches are found, as that would
   * require more than a single swap.
   */
  rep(n) {
    auto ch1 = s1[i], ch2 = s2[i];
    if (ch1 != ch2) {
      st.push_back(i);
    }
    if (st.size() > 2) {
      cout << "NO\n";
      return 0;
    }
  }

  /**
   * @brief Final validation:
   * 1. There must be exactly two differing positions.
   * 2. The characters at these positions must be swappable to make the strings equal.
   *    For differing indices i and j, this means s1[i] must equal s2[j] and s1[j] must equal s2[i].
   */
  if ((st.size() != 2) ||
      !((s1[st[0]] == s2[st[1]]) && (s1[st[1]] == s2[st[0]]))) {
    cout << "NO\n";
    return 0;
  }

  cout << "YES\n";
  return 0;
}