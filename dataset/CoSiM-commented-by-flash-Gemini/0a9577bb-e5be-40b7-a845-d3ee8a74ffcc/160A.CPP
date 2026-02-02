/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - Twins.
 *
 * This problem involves two twins splitting a collection of coins. Each coin has a value.
 * The goal is to find the minimum number of coins one twin needs to take such that
 * the sum of values of their coins is strictly greater than the sum of values of the
 * remaining coins (taken by the other twin).
 *
 * Algorithm: Greedy approach.
 * The code first calculates the total sum of all coins. Then, it iteratively tries
 * to collect coins starting from the largest available value. It maintains a `money`
 * counter (sum of collected coins) and a `counter` for the number of coins taken.
 * The `maximum` variable represents the current largest coin value being considered.
 * It decrements `maximum` and re-scans for coins of that value until `money` exceeds
 * half of the `total` sum.
 *
 * Time Complexity: O(C * MaxVal), where C is the number of coins and MaxVal is the
 * maximum coin value. This is because the outer loop iterates `MaxVal` times in the
 * worst case (e.g., all coins are 1, and `maximum` starts high), and the inner loop
 * iterates `C` times. A more efficient approach would sort the coins and iterate once,
 * achieving O(C log C).
 * Space Complexity: O(C) for storing coin values in an array.
 */

#include <iostream> // For standard C++ input/output operations (cin, cout).

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Twins problem.
 * Functional Utility: Reads the number of coins and their values, then determines
 * the minimum number of coins a twin needs to take to get a sum strictly greater
 * than half of the total sum.
 * Algorithm: Greedy selection of coins, starting from the largest available value.
 */
int main() {
  int total = 0; // Stores the sum of all coin values.
  int maximum = 0; // Stores the current maximum coin value being considered.
  int money = 0; // Stores the sum of coins collected by the current twin.
  int counter = 0; // Stores the number of coins collected by the current twin.
  int flag = 0; // Flag to indicate if a solution has been found (1 = found, 0 = not found).
  int coins; // Stores the total number of coins.
  
  cin >> coins; // Reads the number of coins.
  int value[coins]; // Declares a dynamic array to store coin values.
  
  // Block Logic: Read coin values and calculate total sum and initial maximum value.
  // Precondition: `coins` is a positive integer.
  // Invariant: `total` accumulates the sum, `maximum` stores the highest coin value seen so far.
  for (int i = 0; i < coins; i++) {
    cin >> value[i]; // Reads each coin value.
    total = total + value[i]; // Adds the value to the total sum.
    if (value[i] > maximum) { // Checks if the current coin value is greater than the current maximum.
      maximum = value[i]; // Updates `maximum` if a larger coin is found.
    }
  }
  
  // Block Logic: Greedily collect coins starting from the largest value until the sum exceeds half the total.
  // Precondition: `total` contains the sum of all coins, `maximum` holds the highest coin value.
  // Invariant: `money` increases, `counter` increases, `maximum` decreases to find smaller coins.
  while (flag == 0) {
    for (int i = 0; i < coins; i++) {
      if (maximum == value[i]) { // Block Logic: Find coins with the current `maximum` value.
        money = money + value[i]; // Add the coin's value to the collected money.
        counter++; // Increment the count of collected coins.
        if (money > (total / 2)) { // Check if the collected money is strictly greater than half the total.
          cout << counter; // Print the minimum number of coins.
          flag = 1; // Set flag to exit outer loop.
          break; // Exit inner loop.
        }
      }
    }
    if (flag == 1) { // Check flag to exit outer loop after a solution is found.
      break;
    }
    maximum--; // Decrement `maximum` to search for smaller coins in the next iteration.
  }
  return 0; // Indicate successful program execution.
}
