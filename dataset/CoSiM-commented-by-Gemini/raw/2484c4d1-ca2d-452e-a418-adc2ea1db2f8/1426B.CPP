/**
 * @file 1426B.CPP
 * @brief Solution for Codeforces Problem 1426B, "Symmetric Matrix".
 * @details This program determines if it's possible to construct an m x m symmetric
 * matrix using an infinite supply of given 2x2 tile types.
 *
 * @algorithm The solution relies on checking two necessary conditions:
 * 1. The matrix dimension 'm' must be even. An m x m matrix made of 2x2 tiles
 *    must have an even 'm' so that its total area (m*m) is divisible by 4.
 * 2. There must be at least one available 2x2 tile that is itself symmetric.
 *    A symmetric matrix requires its diagonal blocks to be symmetric. A 2x2 tile
 *    [[a, b], [c, d]] is symmetric if b == c. If such a tile exists, it can be
 *    used to fill the diagonal of the larger matrix.
 * If both conditions are met, the answer is "YES", otherwise it is "NO".
 *
 * @time_complexity O(T * N), where T is the number of test cases and N is the
 * number of tile types for each case. The core logic involves a single pass
 * through the tile types.
 * @space_complexity O(N * 4) per test case to store the definitions of the N tiles.
 */
#include <iostream>
#include <vector>

/**
 * @brief Overloads the >> operator to simplify reading data into a vector.
 * @tparam T The type of elements in the vector.
 * @param input The input stream.
 * @param v The vector to populate.
 * @return The input stream.
 */
template <typename T>
std::istream &operator>>(std::istream &input, std::vector<T> &v) {
  for (T &a : v)
    input >> a;
  return input;
}

/**
 * @brief Prints the final answer "YES" or "NO".
 * @param v A boolean value where true corresponds to "YES" and false to "NO".
 */
void answer(bool v) {
  constexpr const char *s[2] = {"NO", "YES"};
  std::cout << s[v] << '\n';
}

/**
 * @brief Contains the core logic for solving a single test case.
 * @param t A 2D vector representing the available 2x2 tiles.
 * @param m The dimension of the target symmetric matrix.
 */
void solve(const std::vector<std::vector<int>> &t, size_t m) {
  // Condition 1: The matrix dimension 'm' must be even.
  if (m % 2 != 0)
    return answer(false);

  // Condition 2: Search for at least one symmetric tile.
  // A tile [[a,b],[c,d]] is symmetric if b == c. Here, b is x[1] and c is x[2].
  for (const std::vector<int> &x : t) {
    if (x[1] == x[2])
      return answer(true);
  }

  // If m is even but no symmetric tile was found, it's impossible.
  answer(false);
}

/**
 * @brief Handles the input and output for a single test case.
 */
void test_case() {
  size_t n, m;
  std::cin >> n >> m;

  std::vector<std::vector<int>> t(n, std::vector<int>(4));
  std::cin >> t; // Uses the overloaded operator>> to read the tile data.

  solve(t, m);
}

/**
 * @brief Main entry point. Handles multiple test cases.
 */
int main() {
  size_t t;
  std::cin >> t;

  while (t-- > 0)
    test_case();

  return 0;
}