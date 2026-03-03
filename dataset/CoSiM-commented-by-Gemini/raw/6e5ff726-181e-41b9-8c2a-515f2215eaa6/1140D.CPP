
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

/**
 * @file 1140D.CPP
 * @brief Solution to Codeforces problem 1140D, "Minimum Triangulation".
 * @details This program calculates the minimum cost to triangulate a regular
 * n-sided polygon where vertices are labeled 1 to n. The cost of a triangle
 * is the product of its vertex labels, and the total cost is the sum of the
 * costs of all triangles in the triangulation.
 *
 * Algorithm:
 * To minimize the sum of products, the smallest vertex labels should be used
 * as frequently as possible. The optimal strategy is a "fan triangulation"
 * from vertex 1. This involves creating `n-2` triangles by connecting vertex 1
 * to all other non-adjacent vertices. The resulting triangles are (1, 2, 3),
 * (1, 3, 4), ..., (1, n-1, n).
 *
 * The cost of this triangulation is the sum of the products of the vertex
 * labels for each triangle:
 * Cost = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n)
 * This can be written as the summation:
 * Cost = sum_{i=2}^{n-1} (1 * i * (i+1))
 *
 * The code implements this by looping from i = 3 to n and summing `i * (i-1)`,
 * which is equivalent to the formula above.
 *
 * Time Complexity: O(N), for the single loop from 3 to N.
 * Space Complexity: O(1).
 */

#define rep(i, l, r) for (int i = (l); i <= (r); ++i)
#define per(i, l, r) for (int i = (l); i >= (r); --i)
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::make_pair;
using std::pair;
typedef pair<int, int> pii;
typedef long long ll;
typedef unsigned int ui;
int main() {
    // Fast I/O setup
    std::ios::sync_with_stdio(false);
    cout.tie(0);

    // n: number of vertices in the regular polygon.
    int n;
    cin >> n;

    // ans: stores the total minimum cost.
    ll ans = 0;

    /**
     * @brief Calculate the sum of costs for the fan triangulation from vertex 1.
     * The loop calculates sum_{i=3 to n} i * (i-1), which is the cost of
     * triangles (1,2,3), (1,3,4), ..., (1, n-1, n).
     */
    rep(i, 3, n) {
        ans += (ll)i * (i - 1);
    }
    cout << ans;
    return 0;
}
