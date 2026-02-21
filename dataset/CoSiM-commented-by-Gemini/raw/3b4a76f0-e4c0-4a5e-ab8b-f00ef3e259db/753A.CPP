/**
 * @file 753A.CPP
 * @brief Solution to "Santa Claus and Candies" (Codeforces 753A).
 *
 * @details The problem is to represent the integer `n` as a sum of the maximum
 * possible number of distinct positive integers. This corresponds to finding
 * the largest `m` such that `1 + 2 + ... + m <= n`.
 *
 * @algorithm The solution uses a mathematical formula derived from the properties of
 * arithmetic series.
 * 1. The core task is to find the largest integer `m` satisfying the inequality
 *    `m * (m + 1) / 2 <= n`.
 * 2. This can be rewritten as a quadratic inequality `m^2 + m - 2n <= 0`.
 * 3. The positive root of the corresponding equation is `(-1 + sqrt(1 + 8n)) / 2`.
 *    The program calculates the floor of this value to find `m` directly in O(1).
 * 4. This `m` is the maximum number of distinct summands (children).
 * 5. To construct the summands, the program gives `1, 2, ..., m-1` candies to
 *    the first `m-1` children. The last child receives all the remaining candies,
 *    `n - (sum of first m-1 integers)`, which is guaranteed to be a distinct
 *    number greater than `m-1`.
 *
 * @complexity
 * Time Complexity: O(sqrt(N)), dominated by the loop to print the `m` summands,
 *                  where `m` is approximately `sqrt(2N)`.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
// The following are mostly unused macros from a competitive programming template.
#define endl '\n'
#define ll long long int
#define loop(i, a, b) for (ll i = a; i <= b; ++i)
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define clr(x) x.clear()
#define MOD 1000000007
#define itoc(c) ((char)(((int)'0') + c))
#define vl vector<ll>
#define SZ(x) (x).size()
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define sv()                                                                   \
  ll t, n;                                                                     \
  scanf("%lld", &t);                                                           \
  n = t;                                                                       \
  while (t--)
#define tcase()                                                                \
  ll t, n;                                                                     \
  cin >> t;                                                                    \
  n = t;                                                                       \
  while (t--)
#define iscn(num) scanf("%d", &num);
using namespace std;
// Unused template variables and functions.
bool file = 0, rt = 1;
clock_t tStart;
void FAST_IO();
////////////////////////

int main() {

  FAST_IO();
  ////////////////////////
  ll n;
  cin >> n;
  
  // Calculate the maximum number of distinct summands 'm' using the quadratic formula.
  ll m = (sqrt(8 * n + 1) - 1) / 2;
  cout << m << endl;
  
  // Print the first m-1 summands (1, 2, ..., m-1).
  for (int i = 1; i < m; i++)
    cout << i << " ", n -= i;
  
  // The last summand is the remainder of n to ensure the total sum is correct.
  cout << n << endl;

  ////////////////////////
  if (rt && file) {
    printf("\nTime taken: %.6fs", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  }
  return 0;
}

// Unused function for I/O optimization and file handling from a template.
void FAST_IO() {
  ios_base::sync_with_stdio(0);
  // cin.tie(NULL);
  // cout.tie(NULL);
  // cout.setf(ios::fixed);
  // cout.precision(2);
  if (rt && file) {
    tStart = clock();
  }
  if (file) {
#ifndef _offline
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
  }
}
