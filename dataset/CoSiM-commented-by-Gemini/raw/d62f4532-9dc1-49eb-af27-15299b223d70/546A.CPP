/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier must borrow to buy 'w' bananas,
 * where the cost of each banana increases linearly. This implementation wraps the logic in a
 * loop to handle multiple test cases, although the problem on Codeforces only provides one.
 *
 * @algorithm Iterative Summation
 * @complexity
 * Time: O(W), where W is the number of bananas, due to the inner while loop.
 * Space: O(1), using a fixed amount of memory.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries
using namespace std;

int main() {
  int k, n, w;

  // This loop is structured to handle multiple test cases, though only one is expected.
  while (cin >> k >> n >> w) {
    long long sum = 0; // Use long long for safety, though int might pass.
    int count = 1;

    /**
     * @brief Inner loop to calculate the total cost of 'w' bananas.
     * The cost of the 'count'-th banana is count * k.
     */
    while (count <= w) {
      sum += (count * k);
      count++;
    }

    /**
     * @brief Check if the soldier has enough money and print the result.
     * If the total cost 'sum' is less than or equal to the soldier's money 'n',
     * he needs to borrow 0. Otherwise, he borrows the difference.
     */
    if (sum <= n)
      cout << "0" << endl;
    else
      cout << sum - n << endl;
  }
  return 0; // The return is implicit, but explicit is good practice.
}