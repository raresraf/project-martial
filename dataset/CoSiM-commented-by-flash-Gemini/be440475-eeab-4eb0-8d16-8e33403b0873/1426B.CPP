/**
 * @file 1426B.CPP
 * @brief This program solves the "Symmetric Matrix" problem (Codeforces 1426B).
 *   It determines if it's possible to construct an `N x M` matrix by tiling with `N`
 *   given `2x2` matrices, such that the resulting `N x M` matrix is symmetric.
 *   Problem description: https://codeforces.com/contest/1426/problem/B
 * * Algorithm: Conditional check based on matrix dimension parity and `2x2` block symmetry.
 *   1. If `M` (the width of the target matrix) is odd, it's impossible to form a symmetric
 *      matrix using `2x2` blocks, so the answer is "NO".
 *   2. If `M` is even, the program checks if at least one of the `N` provided `2x2` matrices
 *      has internal diagonal symmetry (i.e., its top-right element equals its bottom-left element).
 *   3. If `M` is even and such a symmetric `2x2` block exists, the answer is "YES"; otherwise, "NO".
 * Time Complexity: O(N) where N is the number of `2x2` matrices, as it iterates through them once.
 * Space Complexity: O(1) for a few integer variables and the small `a[4]` array.
 */
// Author - Soumik Roy
// Date & Time - October 02, 2020 12:14 PM
// Problem name - B. Symmetric Matrix
// Problem url - https://codeforces.com/contest/1426/problem/B
// Time limit - 1000 ms
// Memory Limit - 256 MB
// Parsed with ai-virtual-assistant
// https://github.com/Saurav-Paul/AI-virtual-assistant-python

#include <bits/stdc++.h> // Includes most standard C++ libraries.
#define ll long long int // Functional Utility: Alias for long long int.
#define pb push_back     // Functional Utility: Alias for push_back method of vectors.
#define F first          // Functional Utility: Alias for the first element of a pair.
#define S second         // Functional Utility: Alias for the second element of a pair.
#define mp make_pair     // Functional Utility: Alias for std::make_pair.
#define MOD 1000000007   // Constant for modulo operations.
#define vi vector<int>   // Functional Utility: Alias for vector of integers.
#define vll vector<ll>   // Functional Utility: Alias for vector of long longs.
#define pll pair<ll, ll> // Functional Utility: Alias for pair of long longs.
#define pii pair<int, int> // Functional Utility: Alias for pair of integers.
#define all(p) p.begin(), p.end() // Functional Utility: Macro for getting iterators to the beginning and end of a container.
#define mid(s, e) (s + (e - s) / 2) // Functional Utility: Macro to calculate the middle index.
#define eb emplace_back // Functional Utility: Alias for emplace_back method of vectors.
#define ull unsigned long long // Functional Utility: Alias for unsigned long long.
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl; // Debugging macro.
#define KAMEHAMEHA ios_base::sync_with_stdio(0); // Functional Utility: Optimizes C++ standard I/O streams (part 1).
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0); // Functional Utility: Optimizes C++ standard I/O streams (part 2).
using namespace std;
// `dir`: Array representing directions, typically for grid traversal problems (unused in this solution).
int dir[] = {0, 1, 0, -1, 1, 0, -1, 0};

/**
 * @brief Solves a single test case of the Symmetric Matrix problem.
 * Functional Utility: Reads matrix dimensions and 2x2 blocks, then determines
 *                     if a symmetric NxM matrix can be formed.
 */
void solve() {
  int n, m; // `n`: number of 2x2 blocks, `m`: dimension of the target matrix.
  // Block Logic: Reads the number of 2x2 blocks (`n`) and the target matrix dimension (`m`).
  cin >> n >> m;
  bool checkeq = false; // Flag to track if at least one symmetric 2x2 block is found.

  /**
   * Block Logic: Iterates through each of the `n` 2x2 matrices.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `checkeq` becomes true if any 2x2 block has the required symmetry.
   */
  for (int i = 0; i < n; i++) {
    int a[4], x = 0; // `a`: stores the 4 elements of a 2x2 matrix, `x`: unused.
    // Block Logic: Reads the 4 elements of the current 2x2 matrix.
    for (int j = 0; j < 4; j++) {
      cin >> a[j];
    }
    // Conditional Logic: Checks for internal diagonal symmetry in the 2x2 block.
    // For a 2x2 matrix [[a[0], a[1]], [a[2], a[3]]], symmetry requires a[1] == a[2].
    if (a[1] == a[2]) {
      checkeq = true; // Sets the flag if a symmetric 2x2 block is found.
    }
  }
  // Conditional Logic: If `m` is odd, it's impossible to form a symmetric matrix using 2x2 blocks.
  if (m % 2) {
    cout << "NO" << endl;
    return; // Exits the function for this test case.
  }
  // cout<<m<<" "<<checkeq<<endl; // Commented out debugging output.
  // Conditional Logic: If no symmetric 2x2 block was found, it's impossible to form a symmetric matrix.
  if (!checkeq) {
    cout << "NO" << endl;
    return; // Exits the function for this test case.
  }
  // If `m` is even and at least one symmetric 2x2 block was found, print "YES".
  cout << "YES" << endl;
}
/**
 * @brief Main function to handle multiple test cases.
 * Functional Utility: Configures I/O optimization, reads the number of test cases,
 *                     and calls `solve()` for each test case.
 */
int main() {
  KAMEHAMEHA // Applies I/O optimization.
  int t = 1; // `t`: Stores the number of test cases. Default to 1.
  // Block Logic: Reads the actual number of test cases.
  cin >> t;
  /**
   * Block Logic: Iterates through each test case.
   * Pre-condition: `t` is a positive integer.
   * Invariant: `solve()` is called for each test case.
   */
  for (int cn = 1; cn <= t; cn++) {
    solve(); // Calls the function to solve the current test case.
  }
  return 0;
}
