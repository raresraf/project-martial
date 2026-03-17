/**
 * @file 1426B.CPP
 * @author Soumik Roy
 * @brief Solution to Codeforces Problem 1426B - Symmetric Matrix.
 * @date 2020-10-02
 *
 * @details
 * This program determines if it's possible to construct an n x n symmetric matrix
 * composed of smaller m x m blocks, using a given set of 2x2 tiles.
 *
 * @algorithm
 * The problem simplifies to checking two necessary conditions:
 * 1. The sub-block size `m` must be even. If `m` is odd, it's impossible to
 *    form the symmetric structure perfectly, so the answer is "NO".
 * 2. There must be at least one symmetric 2x2 tile available. A tile is
 *    symmetric if its top-right element equals its bottom-left element. Such
 *    a tile is needed for the diagonal blocks of the final matrix to maintain
 *    overall symmetry.
 * If both conditions are met, it is possible to construct such a matrix, and the
 * answer is "YES". Otherwise, it's "NO".
 *
 * @complexity
 * - Time Complexity: O(N) per test case, where N is the number of available tiles.
 *   The program iterates through the tiles once to check for a symmetric one.
 * - Space Complexity: O(1) per test case, as it only stores a few variables and
 *   one tile at a time.
 */

#include <bits/stdc++.h>
#define ll long long int
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define MOD 1000000007
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define eb emplace_back
#define ull unsigned long long
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl;
#define KAMEHAMEHA ios_base::sync_with_stdio(0);
#define RASENGAN                                                               
  ios_base::sync_with_stdio(0);                                                
  cin.tie(0);                                                                  
  cout.tie(0);
using namespace std;
int dir[] = {0, 1, 0, -1, 1, 0, -1, 0};

/**
 * @brief Solves a single test case.
 */
void solve() {
  int n, m;
  cin >> n >> m;
  
  // Flag to track if we have found at least one symmetric 2x2 tile.
  bool checkeq = false;

  /**
   * @brief Read the n tiles and check for a symmetric one.
   * A tile [[a,b],[c,d]] is symmetric if b == c. Here a[1] is b and a[2] is c.
   */
  for (int i = 0; i < n; i++) {
    int a[4], x = 0;
    for (int j = 0; j < 4; j++) {
      cin >> a[j];
    }
    if (a[1] == a[2]) {
      checkeq = true;
    }
  }

  /**
   * @brief First condition check: `m` must be an even number.
   * If `m` is odd, it's impossible to tile symmetrically.
   */
  if (m % 2) {
    cout << "NO" << endl;
    return;
  }
  
  /**
   * @brief Second condition check: A symmetric tile must exist.
   * If we haven't found any tile where the top-right equals the bottom-left,
   * we cannot form the diagonal blocks symmetrically.
   */
  if (!checkeq) {
    cout << "NO" << endl;
    return;
  }

  // If both conditions are met, it is possible.
  cout << "YES" << endl;
}

/**
 * @brief Main function, serves as a driver for the test cases.
 */
int main() {
  RASENGAN // Fast I/O macro
  int t = 1;
  cin >> t;
  for (int cn = 1; cn <= t; cn++) {
    solve();
  }
  return 0;
}
