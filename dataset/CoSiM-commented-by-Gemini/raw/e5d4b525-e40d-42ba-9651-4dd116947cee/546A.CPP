/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 * @details A soldier wants to buy a specific number of bananas. The cost of each
 * banana increases in an arithmetic progression. This program calculates the total
 * cost and determines how much money the soldier needs to borrow.
 *
 * @section
 * Algorithm:
 * 1. Read the cost of the first banana (k), the soldier's initial money (n), and the number of bananas to buy (w).
 * 2. Calculate the total cost required to buy 'w' bananas. The cost of the i-th banana is i*k.
 *    This is equivalent to the sum of an arithmetic series: k * (1 + 2 + ... + w).
 * 3. The total cost is computed iteratively using a loop from 1 to w.
 * 4. Compare the total cost with the soldier's initial money.
 * 5. If the total cost exceeds the initial money, the difference is the amount to be borrowed.
 * 6. If the soldier has enough money, the amount to borrow is 0.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(w), where 'w' is the number of bananas. The cost is calculated using a loop that runs 'w' times.
 * - Space Complexity: O(1), as the program only uses a fixed number of integer variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k, n, w from the problem statement.
  int costOfSingleBanana, totalCost = 0, numOfBanana, initialMoney,
                          moneyBorrowed;

  // Read the initial parameters.
  cin >> costOfSingleBanana >> initialMoney >> numOfBanana;

  /**
   * Block Logic: Calculates the total cost for 'numOfBanana' bananas.
   * The cost for each banana 'i' is 'i * costOfSingleBanana', forming an
   * arithmetic progression.
   * Invariant: At the end of each iteration 'i', 'totalCost' holds the sum of prices for 'i' bananas.
   */
  for (int i = 1; i <= numOfBanana; i++) {
    totalCost = totalCost + (i * costOfSingleBanana);
  }
  
  /**
   * Block Logic: Determine the amount of money to borrow.
   * Pre-condition: 'totalCost' contains the full price of all bananas.
   * This block checks if the soldier's initial money is sufficient.
   */
  if (totalCost > initialMoney) {
    // If the cost is higher than the available money, calculate the deficit.
    moneyBorrowed = totalCost - initialMoney;
    cout << moneyBorrowed << endl;
  } else
    // If the soldier has enough or more money than required, no borrowing is needed.
    cout << 0 << endl;

  return 0;
}