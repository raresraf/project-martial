/**
 * @file 1426B.CPP
 * @brief Solves the "Symmetric Matrix" problem from Codeforces.
 * @details The problem asks if it's possible to construct a symmetric `m x m` matrix using a given set of `n` 2x2 tiles.
 * A key insight for this problem is that to tile a symmetric matrix, the main diagonal of the large matrix must be
 * formed by 2x2 tiles that are themselves symmetric (i.e., for a tile [[a,b],[c,d]], b must equal c).
 * Additionally, an `m x m` matrix can only be tiled by 2x2 blocks if `m` is even.
 * This code attempts to solve the problem by checking these conditions.
 *
 * @algorithm
 * 1. Read the number of test cases, `t`.
 * 2. For each test case, read `n` (number of tiles) and `m` (matrix size).
 * 3. Read `n` tiles. During this process, check if any tile is internally symmetric (`b == c`).
 *    Set a flag `found2` if such a tile exists.
 * 4. The code then enters a complex nested loop to find pairs of tiles that are transposes of each other,
 *    setting a flag `found1` if found. This part of the logic is more complex than required by the problem statement.
 * 5. Finally, it checks if `m` is even. If not, the answer is "NO".
 * 6. If `m` is even, it checks the flags `found1` and `found2` to determine the final "YES" or "NO" answer.
 *
 * @complexity
 * Time complexity: O(T * N^2), where T is the number of test cases and N is the number of tiles. The N^2 factor comes from the nested loop checking for `found1`. A more optimal solution would be O(T * N).
 * Space complexity: O(N) to store the tiles in a vector.
 */
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define N 300005
using namespace std;
typedef long long ll;
ll mod = 1000000007LL;
ll mod2 = 998244353LL;
int t, n, m;
vector<pair<pair<int, int>, pair<int, int>>> tiles;

int main() {
  cin >> t;
  for (int cas = 1; cas <= t; ++cas) {
    scanf("%d%d", &n, &m);
    bool found1 = false; // Flag for finding a specific pair of transposed tiles.
    bool found2 = false; // Flag for finding a single symmetric tile.
    tiles.clear();
    
    /**
     * @brief Reads `n` tiles and checks for any symmetric tile.
     * A symmetric tile ([[a,b],[c,d]]) is one where b == c.
     */
    for (int i = 1; i <= n; ++i) {
      int a, b, c, d;
      scanf("%d%d%d%d", &a, &b, &c, &d);
      tiles.push_back(make_pair(make_pair(a, b), make_pair(c, d)));
      if (b == c) {
        found2 = true;
      }
    }
    
    /**
     * @brief Unnecessary and complex search for a specific relationship between two tiles.
     * This block is not required for a correct solution and adds N^2 complexity.
     */
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int a1 = tiles[i].first.first;
        int b1 = tiles[i].first.second;
        int c1 = tiles[i].second.first;
        int d1 = tiles[i].second.second;
        int a2 = tiles[j].first.first;
        int b2 = tiles[j].first.second;
        int c2 = tiles[j].second.first;
        int d2 = tiles[j].second.second;
        if (c1 == b2 && b1 == c2 && a1 == a2 && d1 == d2) {
          found1 = true;
          break;
        }
      }
      if (found1)
        break;
    }
    
    // A symmetric m x m matrix can only be formed if m is even.
    if (m % 2) {
      cout << "NO" << endl;
    } else {
      // The correct condition is simply the existence of a symmetric tile (`found2`).
      // This implementation uses a more complex set of checks.
      if (m <= 2) { // Special case for m=2
        if (found2)
          cout << "YES" << endl;
        else
          cout << "NO" << endl;
      } else {
        if (found1 && found2) // Overly strict check for m > 2
          cout << "YES" << endl;
        else
          cout << "NO" << endl;
      }
    }
  }
  return 0;
}