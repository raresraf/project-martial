/**
 * @file 1140D.CPP
 * @brief Solution to Codeforces Problem 1140D: "Minimum Triangulation".
 * Calculates the minimum cost to triangulate a regular n-gon.
 * The cost of a triangle is the product of its vertex labels (a*b*c).
 * Algorithm: For a regular n-gon with vertices 1 to n, the minimum cost triangulation
 * involves fixing vertex 1 and forming triangles with adjacent pairs of other vertices:
 * (1, 2, 3), (1, 3, 4), ..., (1, n-1, n). The cost of each such triangle (1, i, i+1)
 * is 1 * i * (i+1). The total minimum cost is the sum of these products.
 * The loop calculates sum(1 * k * (k+1)) for k from 2 to n-1.
 * Time Complexity: O(N) due to a single loop from 3 to N.
 * Space Complexity: O(1) as only a few variables are used.
 */
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
#define rep(i, l, r) for (int i = (l); i <= (r); ++i) // Macro for 'for' loop: iterates from l to r (inclusive).
#define per(i, l, r) for (int i = (l); i >= (r); --i) // Macro for 'for' loop: iterates from l to r (inclusive, decrementing).
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::make_pair;
using std::pair;
typedef pair<int, int> pii; // Type alias for pair of integers.
typedef long long ll;      // Type alias for long long.
typedef unsigned int ui;   // Type alias for unsigned int.

/**
 * @brief Main function to solve the Minimum Triangulation problem.
 * Reads the number of vertices 'n', calculates the minimum triangulation cost, and prints it.
 */
int main() {
    std::ios::sync_with_stdio(false); // Functional Utility: Unties C++ standard streams from C standard streams for faster I/O.
    cout.tie(0);                     // Functional Utility: Prevents cout from flushing before cin, further speeding up I/O.

    int n; // Represents the number of vertices of the regular n-gon.
    cin >> n;
    ll ans = 0; // Stores the accumulated minimum cost of triangulation.

    /**
     * Block Logic: Calculates the minimum cost of triangulating a regular n-gon.
     * The optimal strategy involves fixing vertex 1 as a common vertex for all triangles.
     * The triangles formed are (1, k, k+1) for k ranging from 2 to n-1.
     * The cost of such a triangle (1, k, k+1) is 1 * k * (k+1).
     * The loop iterates with 'i' from 3 to 'n'.
     * When 'i' is the largest vertex in a triangle (1, i-1, i), the product of vertices is 1 * (i-1) * i.
     * Invariant: 'ans' accumulates the sum of products of vertices for the optimal triangulation.
     */
    rep(i, 3, n) {
        ans += (ll)i * (i - 1); // Adds the cost of the triangle formed by vertices (1, i-1, i).
                               // The expression (i * (i-1)) effectively calculates the product of the two larger vertices,
                               // assuming the smallest vertex is 1 and its product contribution is implicit.
    }
    cout << ans; // Prints the total minimum cost.
    return 0;
}
