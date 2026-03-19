/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 *
 * This problem states that a body is in equilibrium if the sum of all forces
 * acting on it is the zero vector. Given `n` force vectors, each with `x, y, z`
 * components, the task is to determine if the body is in equilibrium.
 *
 * Algorithm: Initialize three accumulators for the x, y, and z components of forces to zero.
 * Read each force vector, adding its components to the respective accumulators.
 * After processing all forces, check if all three accumulated sums are zero.
 *
 * Time Complexity: O(N), where N is the number of force vectors. Each force vector
 *                  is processed in constant time.
 * Space Complexity: O(1) as only a few variables are used to store sums.
 */
#pragma comment(linker, "/STACK:16777216") // Pragma directive, often used in competitive programming
                                          // to increase stack size on Windows compilers.
#include <bits/stdc++.h> // Includes various standard libraries for C++, providing common utilities.

using namespace std;

/** @def ms(ar, a)
 * @brief Macro for `memset`, sets all bytes of `ar` to value `a`.
 */
#define ms(ar, a) memset(ar, a, sizeof(ar))
/** @def fr(i, j, k)
 * @brief Loop macro: for `i` from `j` up to `k` (exclusive).
 */
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
/** @def db(x)
 * @brief Debug macro to print a variable's name and value to standard output.
 */
#define db(x) cout << (#x) << " = " << x << endl;
/** @def pb
 * @brief Alias for `push_back`, commonly used with `std::vector`.
 */
#define pb push_back
/** @def mp
 * @brief Alias for `make_pair`, commonly used with `std::pair`.
 */
#define mp make_pair
/** @def X
 * @brief Alias for `first`, commonly used with `std::pair`.
 */
#define X first
/** @def Y
 * @brief Alias for `second`, commonly used with `std::pair`.
 */
#define Y second

/** @typedef ll
 * @brief Type alias for `long long`, used for integers that might exceed the range of `int`.
 */
typedef long long ll;
/** @typedef pii
 * @brief Type alias for `std::pair<int, int>`, commonly used for pairs of integers.
 */
typedef pair<int, int> pii;

/**
 * @brief Converts any given type `_T` to its string representation.
 * @tparam _T The type of the input value.
 * @param a The value to convert to a string.
 * @return The string representation of the input value.
 */
template <class _T> inline string tostr(const _T &a) {
  ostringstream os(""); // Creates an output string stream.
  os << a;              // Writes the value to the stream.
  return os.str();      // Returns the string from the stream.
}

int n; /**< @brief Global variable to store the number of force vectors. */

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Reads the number of force vectors and their components, calculates the
 * resultant force vector, and determines if the body is in equilibrium.
 */
int main() {
  scanf("%d", &n); // Reads the total number of force vectors.
  
  // Initializes accumulators for the x, y, and z components of forces to zero.
  int ans1 = 0, ans2 = 0, ans3 = 0;
  
  /**
   * Block Logic: Iterates through each force vector, summing its components.
   * Pre-condition: `n` specifies the number of force vectors. `ans1`, `ans2`, `ans3` are 0.
   * Invariant: After each iteration, `ans1`, `ans2`, `ans3` hold the accumulated sums of
   *            x, y, z components for the forces processed so far.
   */
  while (n--) { // Loop `n` times.
    int x, y, z; // Temporary variables to read the components of the current force vector.
    scanf("%d%d%d", &x, &y, &z); // Reads the x, y, and z components.
    ans1 += x; // Accumulate x-components.
    ans2 += y; // Accumulate y-components.
    ans3 += z; // Accumulate z-components.
  }

  /**
   * Block Logic: Determines if the body is in equilibrium.
   * The body is in equilibrium if and only if the sum of all x, y, and z
   * components (`ans1`, `ans2`, `ans3`) are each equal to zero.
   * The `||` (OR) operator is used to check if *any* of the sums are non-zero.
   * If any sum is non-zero, `(ans1 || ans2 || ans3)` evaluates to true, indicating "NO" equilibrium.
   * Otherwise (all sums are zero), it evaluates to false, indicating "YES" equilibrium.
   */
  printf("%s\n", (ans1 || ans2 || ans3) ? "NO" : "YES"); // Prints the result based on the equilibrium check.

  return 0; // Indicates successful execution.
}
