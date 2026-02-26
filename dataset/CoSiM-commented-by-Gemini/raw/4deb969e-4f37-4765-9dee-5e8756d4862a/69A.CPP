/**
 * @file 69A.CPP
 * @brief A space-efficient solution for the "Young Physicist" problem.
 * @details This program determines if a body is in equilibrium by checking if the
 * sum of all force vectors acting on it is the zero vector. This implementation
 * is optimized for space by not storing all the vectors.
 *
 * @algorithm
 * Instead of storing all N vectors, this solution processes them on the fly.
 * 1. Initialize a 3-element vector or array (`sum`) to hold the running sums
 *    for the x, y, and z components, all set to zero.
 * 2. Loop N times, once for each vector.
 * 3. In each iteration, read the x, y, and z components of a single vector.
 * 4. Immediately add these components to their respective running sums in the `sum` vector.
 * 5. After the loop, check if all three elements of the `sum` vector are zero.
 *    If so, the net force is zero, and the body is in equilibrium.
 *
 * @complexity
 * Time Complexity: O(N), for a single pass through the N vectors.
 * Space Complexity: O(1), as it only stores the running sums, not the N vectors.
 */
#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Type aliases for convenience, common in competitive programming.
#define ull unsigned long long
#define ll long long

using namespace std;

int main() {
  int n;
  cin >> n;

  // A vector of size 3 to store the sum of x, y, and z components.
  vector<int> sum(3);

  /**
   * @block
   * @brief Process each vector to calculate the component-wise sum.
   * @details This loop reads each vector's components and adds them to the
   * running totals without storing the individual vectors.
   */
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    sum[0] += x; // Accumulate x-components
    sum[1] += y; // Accumulate y-components
    sum[2] += z; // Accumulate z-components
  }

  /**
   * @brief Check for equilibrium and print the result.
   * @details A ternary operator is used for a concise check. If the sum of
   * components along all three axes is zero, the body is in equilibrium.
   */
  cout << (sum[0] == 0 && sum[1] == 0 && sum[2] == 0 ? "YES" : "NO");
  return 0;
}
