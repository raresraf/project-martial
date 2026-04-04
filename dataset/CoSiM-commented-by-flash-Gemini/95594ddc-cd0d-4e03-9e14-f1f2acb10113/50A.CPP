/**
 * @file 50A.CPP
 * @brief Computes the minimum number of 2x1 dominoes required to cover an MxN chessboard.
 * * Algorithm: Ceiling division.
 * Time Complexity: $O(1)$ - constant time arithmetic operations.
 * Space Complexity: $O(1)$ - constant space for variables.
 *
 * This solution addresses the problem of placing the maximum number of 2x1 dominoes
 * on an MxN board without overlap, where M and N are the dimensions of the board.
 * Each domino covers two squares. The problem is equivalent to finding (M*N)/2.
 * If the total number of squares (M*N) is odd, one square will be left uncovered,
 * so the number of dominoes is simply the floor of (M*N)/2. The given code
 * calculates ceil(k/n), which is equivalent to (M*N)/2 in the context of dominoes.
 */

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
// Standard competitive programming macros and constants. While included,
// most are not directly utilized in this specific problem's compact solution.
#define ten5 100000 + 10
#define MOD 1000000007
#define rep(i, a, n) for (int i = a; i < n; i++)
#define iif(c, t, f) ((c) ? (t) : (f))
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define mp make_pair

// Custom comparison function for integers, typical in C-style qsort, but unused here.
int intcmp(const void *v1, const void *v2) { return *(int *)v1 - *(int *)v2; }

ll n, k; // Variables likely representing the total area (k) and domino area (n) if n is 2.

int main() {
  // Functional Utility: Reads two long long integers, 'n' and 'k', from standard input.
  cin >> n >> k;
  // Block Logic: Computes the ceiling of k divided by n.
  // Inline: (k / n) performs integer division (floor).
  // Inline: (k % n) calculates the remainder.
  // Inline: !!(k % n) converts the remainder to a boolean (0 if remainder is 0, 1 if non-zero).
  // This effectively adds 1 if there is any remainder, thus computing ceil(k/n).
  // In the context of dominoes on a chessboard, this calculates (M*N)/2,
  // where M*N is the total area, and n=2 is the area covered by one domino.
  cout << (k / n) + !!(k % n) << endl;

  return 0;
}
