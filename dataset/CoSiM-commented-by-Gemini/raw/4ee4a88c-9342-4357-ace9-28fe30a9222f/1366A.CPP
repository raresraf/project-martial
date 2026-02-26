/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem from Codeforces.
 *
 * @details The problem asks for the maximum number of composite items (emeralds)
 * that can be created from a given number of shovels (`a`) and swords (`b`).
 * One emerald requires either (2 shovels, 1 sword) or (1 shovel, 2 swords).
 *
 * @algorithm
 * This is a resource allocation problem. The maximum number of items `k` is
 * constrained by the total number of each resource and the combined total.
 * 1. `k <= a` (cannot make more items than the number of shovels)
 * 2. `k <= b` (cannot make more items than the number of swords)
 * 3. `3k <= a + b` (each item uses 3 resources in total) -> `k <= (a+b)/3`
 *
 * Combining these, the general solution is `k = min(a, b, (a+b)/3)`.
 * The code implements this logic through case analysis after ensuring `a <= b`.
 * - If `b >= 2a`, the bottleneck is the number of shovels, so the answer is `a`.
 *   This corresponds to the general solution, as `a` will be the minimum value
 *   in `min(a, b, (a+b)/3)`.
 * - If `b < 2a`, the resources are more balanced, and the bottleneck is the
 *   total number of items, so the answer is `(a+b)/3`. The code uses a more
 *   complex formula which simplifies to this.
 *
 * @complexity
 * Time Complexity: O(1) for each test case, as it involves a few comparisons
 * and arithmetic operations.
 * Space Complexity: O(1).
 */
#include <iostream>
#include <algorithm> // For std::swap

/**
 * @brief A helper function to print the answer for a test case.
 */
void answer(unsigned v) { std::cout << v << '\n'; }

/**
 * @brief Contains the core logic to solve for a single pair of a and b.
 * @param a The number of shovels.
 * @param b The number of swords.
 */
void solve(unsigned a, unsigned b) {
  // This initial check is redundant but correct.
  if (a == b)
    return answer((a + b) / 3);

  // Ensure `a` holds the smaller value.
  if (a > b)
    std::swap(a, b);

  /**
   * @brief Case 1: Swords are abundant (b >= 2*a).
   * In this case, the number of shovels `a` is the limiting factor. We can
   * form `a` items of type (1 shovel, 2 swords).
   */
  if (2 * a <= b)
    return answer(a);

  /**
   * @brief Case 2: Resources are relatively balanced (b < 2*a).
   * The limiting factor is the total number of items. The answer is (a+b)/3.
   * The code uses a more complex formula that simplifies to this:
   * Let k = 2a - b.
   * The formula is: 2*k/3 + a - k = (2*(2a-b))/3 + (3*(a-k))/3
   * = (4a - 2b)/3 + a - (2a - b)
   * = (4a - 2b)/3 + b - a
   * = (4a - 2b + 3b - 3a) / 3
   * = (a + b) / 3
   */
  const unsigned k = 2 * a - b;
  answer(2 * k / 3 + a - k);
}

/**
 * @brief Reads inputs for a single test case and calls the solver.
 */
void test_case() {
  unsigned a, b;
  std::cin >> a >> b;
  solve(a, b);
}

/**
 * @brief Main function to handle multiple test cases.
 */
int main() {
  // It's good practice to add I/O optimization for competitive programming.
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);
  
  size_t t;
  std::cin >> t;

  while (t-- > 0)
    test_case();

  return 0;
}