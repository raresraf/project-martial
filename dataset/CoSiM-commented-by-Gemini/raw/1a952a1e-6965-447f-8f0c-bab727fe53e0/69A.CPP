/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @details This program determines if a body is in equilibrium by checking if the
 * vector sum of all forces acting upon it is zero. It uses a common competitive
 * programming template structure with global variables for accumulating sums.
 *
 * @section algorithm Algorithm
 * 1. Three global integer variables (`a`, `b`, `c`) are used to store the sum of
 *    forces along the X, Y, and Z axes, respectively. They are initialized to zero.
 * 2. The `main` function reads an integer `tc`, which represents the number of force
 *    vectors to be processed.
 * 3. A loop runs `tc` times. In each iteration, a `solve()` function is called.
 * 4. The `solve()` function reads the three components (x, y, z) of a single
 *    force vector and adds them to the corresponding global sum variables.
 * 5. After the loop finishes, the `main` function checks if all three global sums
 *    (`a`, `b`, and `c`) are zero.
 * 6. If they are, it prints "YES", indicating the body is in equilibrium.
 *    Otherwise, it prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of force vectors (`tc`), because
 *   the program processes each vector once.
 * - Space Complexity: O(1), as it uses a fixed amount of memory (a few global
 *   variables) regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;

// Typedefs and macros for competitive programming shorthand.
typedef long long ll;
typedef pair<int, int> pii;
// ... (other boilerplate)

#define FAST_INP ios_base::sync_with_stdio(false); cin.tie(NULL)

// Global variables to store the sum of forces along each axis.
int a = 0, b = 0, c = 0;
int x, y, z; // Temporary variables for reading input.

/**
 * @brief Reads a single force vector and adds it to the global sums.
 */
void solve() {
  cin >> x >> y >> z;
  a += x; // Accumulate x-component.
  b += y; // Accumulate y-component.
  c += z; // Accumulate z-component.
}

/**
 * @brief Main function to handle input and check for equilibrium.
 * @return 0 on successful execution.
 */
int main() {
  FAST_INP;
#ifndef ONLINE_JUDGE
  // Redirect I/O for local testing.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int tc; // Number of test cases, which is the number of force vectors.
  cin >> tc;
  
  // Process each force vector by calling solve().
  while(tc--) solve();
  
  // Check if the net force on all axes is zero.
  cout << ((a == 0 && b == 0 && c == 0) ? "YES" : "NO") << endl;
  
  return 0;
}