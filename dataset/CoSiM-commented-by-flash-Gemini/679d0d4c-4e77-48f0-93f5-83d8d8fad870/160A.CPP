/**
 * @679d0d4c-4e77-48f0-93f5-83d8d8fad870/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. It reads all coin values, calculates their total sum,
 * sorts the coins in descending order, and then iteratively takes the largest coins
 * until their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) for storing coin values in a vector.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream>, <vector>, <algorithm>, and <numeric>.
#include <vector>        // Explicitly included for std::vector.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Comparison function for sorting in descending order.
 * Used with `std::sort` to arrange elements from largest to smallest.
 * @param x The first integer for comparison.
 * @param y The second integer for comparison.
 * @return True if x is greater than y, false otherwise.
 */
bool chg(int x, int y) { return x > y; }

/**
 * @brief Main function to execute the Twins problem logic.
 * Reads coin values, calculates total sum, sorts coins, and determines
 * the minimum number of coins needed to exceed half the total sum.
 * @return 0 on successful execution.
 */
int main() {
  vector<int> vect; // Declares a vector to store the coin values.
  int t, n, sum = 0, i, j; // 't' is unused, 'n' is coin value, 'sum' is player's total, 'i', 'j' are loop counters.
  
  // The variable 't' is read here, but based on problem 160A, it's typically 'n' for number of coins.
  // It seems 't' is misused here for 'n', and 'n' is used for individual coin value.
  cin >> t; // Reads the number of coins (misnamed 't').
  
  /**
   * Block Logic: Reads 't' (which should be 'n') coin values and stores them in `vect`.
   * Invariant: `vect` contains all input coin values.
   */
  while (t--) { // Loops 't' times.
    cin >> n; // Reads each coin value (misnamed 'n').
    vect.push_back(n); // Adds the coin value to the vector.
  }
  
  // Sorts the vector `vect` in descending order using the custom comparison function `chg`.
  sort(vect.begin(), vect.end(), chg); 
  
  // Calculates the initial total sum of all coins using `std::accumulate`.
  int p = accumulate(vect.begin(), vect.end(), 0); 

  /**
   * Block Logic: Iterates through the coins (from largest to smallest) and greedily takes them.
   * Accumulates their sum (`sum`) for the current player, and updates the remaining sum (`p`).
   * Stops and prints the count of coins taken when the current player's sum strictly
   * exceeds the sum of the remaining coins.
   * Invariant: `sum` stores the current player's total; `p` stores the remaining total.
   */
  for (i = 0; i < vect.size(); i++) { // Loops through coins in descending order.
    sum += vect[i]; // Adds the current largest coin to the player's sum.
    p = p - vect[i]; // Subtracts the current coin from the remaining sum.
    
    // Pre-condition: Check if the current player's sum is strictly greater than the remaining sum.
    if (sum > p) { 
      cout << i + 1 << endl; // Prints the count of coins taken (i+1 because i is 0-indexed).
      return 0; // Exits the program early once the condition is met.
    }
  }

  return 0; // Indicates successful program termination (should ideally not be reached if N > 0).
}
