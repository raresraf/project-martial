/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy approach.
 *
 * This program finds the minimum number of coins required to make the sum of their
 * values strictly greater than the sum of the remaining coins. It does this by
 * first calculating the total sum of all coins. Then, it sorts the coins and
 * greedily picks the largest ones until their sum exceeds half of the total sum.
 *
 * @author f30f6bf5-4b09-4ca2-9d6a-b0bcec3ef469
 *
 * @algorithm Greedy Algorithm with Sorting
 * @complexity
 * Time: O(A log A), where A is the number of coins, dominated by the sorting step.
 * Space: O(A), to store the coin values in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

// Global variables to store input and state.
int a,      // Number of coins.
    mas[105], // Array to store coin values.
    sum,    // Sum of the coins taken.
    num,    // Number of coins taken.
    jami;   // Total sum of all coins.

int main() {
  // Read the number of coins.
  cin >> a;

  /**
   * @brief First pass: Read coin values and calculate the total sum (`jami`).
   */
  for (int i = 0; i < a; i++) {
    cin >> mas[i];
    jami += mas[i];
  }
  // Sort the coins in non-decreasing order.
  sort(mas, mas + a);

  /**
   * @brief Second pass: Greedily take the largest coins.
   * This loop iterates from the largest coin downwards. It continues as long as
   * the sum of the taken coins (`sum`) is not more than half of the total sum.
   * Once `sum` is greater than `jami / 2`, the condition `val > jami - val` is met.
   */
  for (int i = (a - 1); i >= 0; i--) {
    if (sum <= jami / 2) {
      sum += mas[i]; // Add the largest available coin to our pile.
      num++;         // Increment the count of coins taken.
    }
  }
  // Print the minimum number of coins required.
  cout << num;
}