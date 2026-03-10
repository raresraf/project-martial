/**
 * @file 1140D.CPP
 * @brief Solution for a competitive programming problem (Codeforces 1140D "Minimum Triangulation").
 * This program calculates the minimum cost to triangulate a regular n-sided polygon (n-gon),
 * where the cost of a triangulation is the sum of the products of the vertices of each triangle.
 *
 * NOTE: This file contains many boilerplate macros and functions for competitive
 * programming that are not used in this specific solution.
 *
 * @algorithm The solution is based on the insight that the minimum cost is achieved
 * by fixing one vertex (e.g., vertex 1) and forming a "fan" triangulation. This involves
 * creating triangles (1, 2, 3), (1, 3, 4), ..., (1, n-1, n). The total cost is the
 * sum of the products of these vertices' indices. Since one vertex is always 1,
 * the formula simplifies to the sum of `i * (i+1)` for i from 2 to n-1.
 *
 * @time_complexity O(N), where N is the number of vertices, due to the single loop.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
#define endl '\n'
#define time clock_t tStart = clock();
#define show                                                                   \
  printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
#define ll long long int
#define loop(a, b) for (int i = a; i <= b; ++i)
#define count_1(n) __builtin_popcountll(n)
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define clr(x) x.clear()
#define MOD 1000000007
#define itoc(c) ((char)(((int)'0') + c))
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
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
void FAST_IO();
int main() {
  //_time_
  // FAST_IO();
  ll n, len = 0;
  cin >> n;
  /**
   * @block
   * @brief Calculates the sum of triangulation costs.
   * This loop implements the formula for the minimum triangulation cost:
   * sum_{i=2}^{n-1} i * (i+1). This corresponds to the cost of triangles
   * (1,2,3), (1,3,4), ..., (1,n-1,n), assuming vertex 1 is the fixed pivot.
   */
  for (ll i = 2; i < n; i++) {
    len += (i * (i + 1));
  }
  cout << len << endl;
  // show
  return 0;
}

/**
 * @brief A utility function for boilerplate setup in competitive programming.
 * It optimizes I/O and can redirect streams for local testing.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  // cout.setf(ios::fixed);
  // cout.precision(20);
#ifndef _offline
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);
#endif
}
