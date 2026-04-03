/**
 * @8cc6b86a-bc5b-4220-8d6b-04a03dff16b1/546A.CPP
 * @brief This program calculates the total cost to purchase a specified number of items
 * where the cost of each subsequent item increases linearly. It then determines if
 * additional money is needed given the initial amount. This is a common problem
 * related to arithmetic series sum.
 * Algorithm: Iterative summation of an arithmetic series.
 * Time Complexity: O(W), where W is the number of items to buy, due to the loop.
 * Space Complexity: O(1) for storing input variables and calculations.
 */

#include <algorithm> // Not directly used in this specific logic but often included in competitive programming.
#include <cctype>    // Not directly used.
#include <cmath>     // Not directly used.
#include <cstring>   // Not directly used.
#include <iostream>  // Required for standard input/output operations (cin, cout).
#include <string>    // Not directly used.
using namespace std;

int main() {
  // @var k: The base cost of the first item (the multiplier for subsequent items).
  // @var n: The initial amount of money the person has.
  // @var w: The number of items (bananas) to buy.
  int k, n, w;
  cin >> k >> n >> w; // Read the base cost, initial money, and number of items.
  
  // @var cost: Accumulates the total cost of purchasing 'w' items.
  int cost = 0;
  // Block Logic: Calculate the total cost of buying 'w' items.
  // Invariant: 'cost' accumulates the sum of (i * k) for i from 1 to w.
  // Pre-condition: 'w' is the upper limit for the loop.
  for (int i = 1; i <= w; i++) {
    // Inline: Add the cost of the i-th item (i * k) to the total cost.
    cost = cost + (i * k); // Example: 1*3+2*3+3*3+4*3
  }

  // Block Logic: Determine if additional money is required.
  // If the total calculated cost is greater than the available money, print the difference.
  if (cost > n) {
    cout << cost - n; // Print the amount of money needed.
  } 
  // Else, if available money is sufficient or more, no additional money is needed.
  else {
    cout << 0; // Print 0, indicating no deficit.
  }
  return 0; // Explicitly return 0 for successful execution.
}
