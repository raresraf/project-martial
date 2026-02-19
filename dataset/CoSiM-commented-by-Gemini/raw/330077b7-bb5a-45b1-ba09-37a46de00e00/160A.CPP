/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm with `std::sort`.
 * @details This program finds the minimum number of coins to take from a set so that
 * their sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm The solution employs a greedy strategy. It recognizes that to achieve the
 * target sum with the minimum number of coins, one must always pick the largest
 * available coins. The implementation is as follows:
 * 1. Read all coin values, calculate their total sum, and store them in a vector.
 * 2. Sort the coins in ascending order using `std::sort`.
 * 3. Iterate backwards from the largest coin, accumulating its value.
 * 4. After picking each coin, check if the accumulated sum (`sum1`) is strictly greater
 *    than the sum of the remaining coins. This is efficiently checked with `2 * sum1 > sum`.
 * 5. Once this condition is met, the number of coins taken is the answer, and the loop terminates.
 *
 * @time_complexity O(N log N), where N is the number of coins, dominated by the sorting operation.
 * @space_complexity O(N), to store the coin values in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  int coin_count;
  cin >> coin_count;

  vector<int> vec;
  // `sum` stores the total value of all coins.
  int sum = 0;
  // Block Logic: Read all coin values, calculate the total sum, and store them in a vector.
  for (int i = 0; i < coin_count; i++) {
    int temp;
    cin >> temp;
    sum += temp;
    vec.push_back(temp);
  }

  // `sum1` will store the sum of the coins taken.
  int sum1 = 0;
  // Sort the coins to facilitate the greedy approach of picking the largest ones.
  sort(vec.begin(), vec.end());

  /**
   * Block Logic: Greedily take the largest coins one by one until their sum is a majority.
   * The loop iterates from the end of the sorted vector (largest element) to the beginning.
   */
  for (int i = vec.size() - 1; i >= 0; i--) {
    sum1 += vec[i];

    // Pre-condition: Check if the sum of coins taken (`sum1`) is strictly greater
    // than the sum of the remaining coins (`sum - sum1`). The check `2 * sum1 > sum`
    // is an efficient way to do this without subtraction.
    if (2 * sum1 > sum) {
      // The number of coins taken is the count of iterations from the end of the vector.
      cout << (vec.size() - i);
      break;
    }
  }

  return 0;
}