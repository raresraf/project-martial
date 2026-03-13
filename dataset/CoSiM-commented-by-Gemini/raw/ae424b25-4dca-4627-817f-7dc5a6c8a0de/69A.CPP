/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 *
 * This program determines if a body is in equilibrium under the influence of
 * several forces. A body is in equilibrium if the vector sum of all forces
 * acting on it is zero. The program reads a set of 3D vectors and checks if
 * the sum of their x, y, and z components are all zero.
 *
 * @note The macros used in this file are a common practice in competitive
 * programming for brevity.
 *
 * @see https://codeforces.com/problemset/problem/69/A
 */
#include <bits/stdc++.h>

#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief Reads force vectors and checks for equilibrium.
 *
 * This function reads `n` 3D force vectors. It calculates the sum of the x, y,
 * and z components separately. If all three sums are zero, the body is in
 * equilibrium, and the function prints "YES"; otherwise, it prints "NO".
 */
void solve() {
  int n;
  cin >> n;
  int col1 = 0, col2 = 0, col3 = 0;
  int x, y, z;
  /**
   * @brief Sums the components of the force vectors.
   */
  _for(i, 0, n) {
    cin >> x >> y >> z;
    col1 += x;
    col2 += y;
    col3 += z;
  }
  /**
   * @brief Checks the equilibrium condition.
   */
  if (!col1 && !col2 && !col3)
    cout << "YES";
  else
    cout << "NO";
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
#ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  solve();
}
