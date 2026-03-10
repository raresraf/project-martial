/**
 * @file 1426B.CPP
 * @author Shammi Anand
 * @brief Solution for a competitive programming problem.
 * This program determines if it is possible to construct a symmetric square
 * matrix of size M x M using a given set of N 2x2 matrix tiles.
 *
 * @algorithm The solution relies on two key insights:
 * 1. An M x M matrix can only be perfectly tiled by 2x2 matrices if M is even.
 * 2. To construct a larger symmetric matrix, at least one of the constituent
 *    2x2 tiles must itself be symmetric (i.e., its top-right element equals its
 *    bottom-left element).
 * The code checks for these two conditions.
 *
 * @time_complexity O(N) per test case, where N is the number of tiles, due to the single loop to check for a symmetric tile.
 * @space_complexity O(N) per test case to store the N 2x2 matrices.
 */
/*
This code is written by Shammi Anand
contact : shammianand101@gmail.com, shammianand.me
*/
#include <bits/stdc++.h>
using namespace std;

// Macros for common operations and types
#define ll long long
#define pb push_back
#define f first
#define s second
#define vi vector<int>
#define pii pair<int, int>
#define rep(i, a, n) for (int i = a; i < n; i++)
#define F(i, n) for (int i = 0; i < n; i++)
#define all(a) a.begin(), a.end()
#define INF 1e9 + 7
#define nl "\n"
// Macro for handling multiple test cases
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)

/**
 * @brief A template function for printing vector contents.
 * Used for debugging purposes, not essential to the core logic.
 */
template <typename T_vector>
void output(const T_vector &v, int start = -1, int end = -1) {
  if (start < 0)
    start = 0;
  if (end < 0)
    end = int(v.size());

  for (int i = start; i < end; i++) {
    cout << v[i] << (i < end - 1 ? ' ' : '\n');
  }
}

/**
 * @brief A utility function for boilerplate setup in competitive programming.
 * It optimizes I/O and redirects streams to files for local testing.
 */
void shammi() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
}

int main() {
  shammi();
  w(t) {
    int n, m;
    cin >> n >> m;
    int matrices[n][2][2];
    for (int i = 0; i < n; i++) {
      cin >> matrices[i][0][0] >> matrices[i][0][1] >> matrices[i][1][0] >>
          matrices[i][1][1];
    }
    /**
     * @block
     * @brief Core logic to check for the possibility of a symmetric matrix.
     * First, checks if the target dimension `m` is even. A matrix cannot be
     * tiled by 2x2 blocks if its dimension is odd.
     */
    if (m % 2) {
      cout << "NO\n";
    } else {
      /**
       * @brief Checks if at least one of the provided tiles is symmetric.
       * A symmetric 2x2 tile is required as a building block for the larger
       * symmetric matrix. The condition `matrices[i][0][1] == matrices[i][1][0]`
       * checks for this symmetry.
       */
      bool ok = false;
      for (int i = 0; i < n; i++) {
        if (matrices[i][0][1] == matrices[i][1][0]) {
          ok = true;
          break;
        }
      }
      cout << (ok ? "YES\n" : "NO\n");
    }
  }
  return 0;
}