/**
 * @file 546A.CPP
 * @brief An optimal O(1) solution for the "Soldier and Bananas" problem.
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * 'w' bananas, where the cost increases arithmetically. This implementation
 * directly calculates the total cost using the mathematical formula for the sum
 * of an arithmetic series, which is highly efficient.
 *
 * The total cost is `k * (1 + 2 + ... + w)`, which simplifies to
 * `k * w * (w + 1) / 2`.
 *
 * @section
 * Algorithm: Direct calculation using an arithmetic series summation formula.
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <cstdio>

using namespace std;

int main() {
  // cost: The base cost of the first banana (k in the problem statement).
  // money: The initial amount of money the soldier has (n in the problem statement).
  // row: The number of bananas to buy (w in the problem statement).
  // count: Will hold the calculated total cost.
  int count, row, money, cost;
  scanf("%d %d %d", &cost, &money, &row);

  // This special case for a single banana is handled correctly but is
  // redundant, as the general formula also works for row = 1.
  if (row == 1) {
    if (money > cost)
      printf("0");
    else
      printf("%d", cost - money);
  } else {
    /**
     * @brief Calculate the total cost using a mathematical formula.
     *
     * Logic: The sum of the cost multipliers (1 + 2 + ... + row) is calculated
     * first using the formula for the sum of the first N integers: N * (N + 1) / 2.
     * This sum is then multiplied by the base cost.
     */
    count = (row * (row + 1)) / 2;
    count = count * cost;

    // Determine the amount to borrow by comparing the total cost to available money.
    if (money > count)
      printf("0");
    else
      printf("%d", count - money);
  }
  return 0;
}
