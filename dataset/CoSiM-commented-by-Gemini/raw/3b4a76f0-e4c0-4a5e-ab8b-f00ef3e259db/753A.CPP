/**
 * @file 753A.CPP
 * @brief Distributes n candies among the maximum number of people with distinct shares.
 *
 * This program solves the problem of partitioning an integer `n` (candies) into
 * the maximum possible number of unique positive integer summands. For example,
 * if n=8, the maximum number of people is 3, with shares 1, 2, 5.
 *
 * Algorithm:
 * 1. The problem is equivalent to finding the largest integer `m` such that the
 *    sum of the first `m` integers `1 + 2 + ... + m` is less than or equal to `n`.
 *    The sum of this arithmetic series is `m * (m + 1) / 2`.
 * 2. The program solves the quadratic equation `m^2 + m - 2n = 0` for `m` to find
 *    the maximum number of participants. The positive root is given by the
 *    formula `m = (sqrt(8*n + 1) - 1) / 2`. Integer division handles the floor.
 * 3. The calculated value `m` is printed, which is the maximum number of people.
 * 4. A greedy approach is used to print the shares. The first `m-1` people
 *    receive `1, 2, ..., m-1` candies respectively.
 * 5. The last person (`m`-th person) receives all the remaining candies. The
 *    code calculates this by starting with the total `n` and subtracting each
 *    of the first `m-1` shares. The final value of `n` is the share for the
 *    last person.
 *
 * Time Complexity: O(sqrt(N)), because the loop to print the shares runs `m-1`
 * times, and `m` is approximately `sqrt(2N)`.
 * Space Complexity: O(1).
 *
 * @note This file contains many unused preprocessor macros and a boilerplate
 *       `FAST_IO` function common in competitive programming.
 */
#include <bits/stdc++.h>
// Unused competitive programming macros
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

// Unused boilerplate variables and functions
bool file = 0, rt = 1;
clock_t tStart;
void FAST_IO();
////////////////////////

int main() {

  FAST_IO();
  ////////////////////////
  ll n;
  cin >> n;
  
  // Calculate the maximum number of people (m) using the quadratic formula.
  ll m = (sqrt(8 * n + 1) - 1) / 2;
  cout << m << endl;
  
  /**
   * @brief Greedily assign shares 1, 2, ..., m-1.
   * The loop gives the first m-1 people their shares and subtracts them
   * from the total `n`.
   */
  for (int i = 1; i < m; i++)
    cout << i << " ", n -= i;
    
  // The last person gets all the remaining candies.
  cout << n << endl;

  ////////////////////////
  if (rt && file) {
    printf("\nTime taken: %.6fs", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  }
  return 0;
}

// Unused boilerplate function for fast I/O setup.
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
