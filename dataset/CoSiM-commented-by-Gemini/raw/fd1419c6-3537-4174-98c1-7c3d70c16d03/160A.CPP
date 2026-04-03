/**
 * @file 160A.cpp
 * @brief An optimized solution to the "Twins" competitive programming problem.
 * This program finds the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Optimized Greedy Approach
 * 1. Read the number of coins `N` and their values into a vector.
 * 2. Calculate the total sum of all coin values and divide it by 2 to get a `lim`.
 *    The goal is to find the smallest set of coins with a sum greater than `lim`.
 * 3. Sort the coins in descending order.
 * 4. Greedily take the largest coins one by one, accumulating their sum in `a`.
 * 5. Count the number of coins taken (`i`).
 * 6. Stop when the accumulated sum `a` is greater than `lim`.
 * 7. Print the number of coins taken.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>

int main() {
  int N;
  std::cin >> N;
  std::vector<int> vec(N);
  int lim = 0;
  // Read coin values and calculate the total sum.
  for (int i = 0; i < N; ++i) {
    std::cin >> vec[i];
    lim += vec[i];
  }
  // The target sum to exceed is half of the total sum.
  lim /= 2;
  
  // Sort the coins in descending order.
  std::sort(vec.begin(), vec.end(), std::greater<int>());
  
  int a = 0, i = 0;
  // Greedily take the largest coins until their sum exceeds the target.
  while (a <= lim)
    a += vec[i++];
  
  // Print the number of coins taken.
  std::cout << i << std::endl;

  return 0;
}
