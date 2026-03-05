/**
 * @file 69A.CPP
 * @brief Checks if a system of 3D vectors is in equilibrium.
 *
 * This program reads an integer `n` followed by `n` three-dimensional vectors.
 * It then determines if the sum of these vectors results in a zero vector,
 * which signifies that the system is in equilibrium.
 *
 * @algorithm
 * 1. Read the total number of vectors, `n`.
 * 2. Initialize three variables (`ans1`, `ans2`, `ans3`) to store the sum of the
 *    x, y, and z components of the vectors, respectively. All are initialized to 0.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of the current vector.
 *    b. Add these components to their respective sum variables.
 * 4. After the loop, check if all three sum variables (`ans1`, `ans2`, `ans3`) are zero.
 * 5. If `ans1`, `ans2`, and `ans3` are all zero, the system is in equilibrium, so print "YES".
 * 6. Otherwise, the system is not in equilibrium, so print "NO".
 *
 * @complexity
 * Time Complexity: O(n), because the program iterates through the `n` vectors once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the number of vectors.
 */

// This pragma is a compiler-specific directive (for MSVC) to increase the stack size.
// It's often used in competitive programming to prevent stack overflow with deep recursion.
#pragma comment(linker, "/STACK:16777216")
// Includes all standard C++ libraries.
#include <bits/stdc++.h>

using namespace std;

// Macro for setting array elements to a value.
#define ms(ar, a) memset(ar, a, sizeof(ar))
// Macro for a standard for-loop.
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
// Macro for debugging, prints a variable's name and value.
#define db(x) cout << (#x) << " = " << x << endl;
// Macro for vector push_back.
#define pb push_back
// Macro for creating a pair.
#define mp make_pair
// Macro to access the first element of a pair.
#define X first
// Macro to access the second element of a pair.
#define Y second

// Type alias for long long integers.
typedef long long ll;
// Type alias for a pair of integers.
typedef pair<int, int> pii;

/**
 * @brief Converts a value of any type to a string.
 * @tparam _T The type of the value to convert.
 * @param a The value to convert.
 * @return The string representation of the value.
 */
template <class _T> inline string tostr(const _T &a) {
  ostringstream os("");
  os << a;
  return os.str();
}

// Number of vectors.
int n;

int main() {

  // Read the number of vectors.
  scanf("%d", &n);
  // Initialize sums of x, y, and z components to zero.
  int ans1 = 0, ans2 = 0, ans3 = 0;

  /**
   * @brief Loop to read vectors and compute component sums.
   *
   * Invariant: At the beginning of each iteration, ans1, ans2, and ans3 hold the
   * sums of the components of the vectors read so far.
   */
  while (n--) {
    int x, y, z;
    // Read the three components of a vector.
    scanf("%d%d%d", &x, &y, &z);
    // Add the components to the total sums.
    ans1 += x, ans2 += y, ans3 += z;
  }

  /**
   * @brief Check for equilibrium condition.
   * The condition (ans1 || ans2 || ans3) checks if any of the sums are non-zero.
   * If any sum is non-zero, the expression is true, and it prints "NO".
   * If all sums are zero, the expression is false, and it prints "YES".
   */
  printf("%s\n", (ans1 || ans2 || ans3) ? "NO" : "YES");

  return 0;
}
