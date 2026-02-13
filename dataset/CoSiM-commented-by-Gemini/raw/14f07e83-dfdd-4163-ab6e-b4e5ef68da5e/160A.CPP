/**
 * @file 160A.CPP
 * @brief Efficient solution for the "Twins" problem.
 * @details The program determines the minimum number of coins one needs to take from a set
 * so that the sum of their values is strictly greater than the sum of the values of the
 * remaining coins. This is achieved by greedily taking the coins with the highest values.
 *
 * @section algorithm Algorithm
 * 1. Read the number of coins and their values, calculating the total sum of all coins.
 * 2. Sort the coins in non-decreasing order.
 * 3. Greedily take the largest coins one by one, adding their value to a running sum (`outputValue`)
 *    and counting how many coins have been taken (`outputCoins`).
 * 4. The process stops when the sum of the taken coins (`outputValue`) becomes strictly greater
 *    than half of the total sum. The condition `outputValue > total / 2` is equivalent to
 *    `outputValue > total - outputValue`, which means our sum is larger than the sum of the remaining coins.
 * 5. The final count `outputCoins` is the minimum number of coins required.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N log N), dominated by the sorting step.
 * - Space Complexity: O(N), to store the coin values in a dynamically allocated array.
 */
#include <algorithm>
#include <cstdio>
using namespace std;

int main() {
  int numCoins;
  scanf("%d\n", &numCoins);
  int *coins = new int[numCoins];
  long total = 0, outputCoins = 0, outputValue = 0;
  
  // Read all coin values and calculate the total sum.
  for (int k = 0; k < numCoins; k++) {
    scanf("%d", coins + k);
    total += coins[k];
  }
  
  // Sort the coins in ascending order to easily access the largest values.
  sort(coins, coins + numCoins);

  /**
   * @brief Greedily takes the largest coins until their sum exceeds half of the total.
   * The loop condition `outputValue <= total / 2` ensures that we stop as soon as
   * our sum is strictly greater than the sum of the remaining coins.
   */
  while (outputValue <= total / 2) {
    // Take one more coin.
    ++outputCoins;
    // Add the value of the next largest coin to our running total.
    outputValue += coins[numCoins - outputCoins];
  }
  printf("%ld\n", outputCoins);
  return 0;
}
