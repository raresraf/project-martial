//
// Created by Conor on 5/28/2020.
//

/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A: Twins (Vector-based Optimized Implementation).
 * @details This program solves a problem where it's required to take the minimum number of coins
 * (represented by integers) such that their sum is strictly greater than the sum of the remaining coins.
 * It reads 'n' coin values, calculates their total sum, stores them in a vector, and sorts the coins
 * in descending order. Then, it iteratively sums the largest coins while simultaneously subtracting
 * from the total sum of remaining coins until the sum of taken coins strictly exceeds the sum of remaining coins.
 * The count of these taken coins is then printed.
 * @algorithm Greedy approach with sorting a vector.
 * Time Complexity: $O(N \log N)$ due to sorting, where N is the number of coins.
 * Space Complexity: $O(N)$ to store the coin values in a vector.
 */

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

//

// Created by Conor on 5/28/2020.

//



/**

 * @file 160A.CPP

 * @brief Solution for Codeforces Problem 160A: Twins (Vector-based Optimized Implementation).

 * @details This program solves a problem where it's required to take the minimum number of coins

 * (represented by integers) such that their sum is strictly greater than the sum of the remaining coins.

 * It reads 'n' coin values, calculates their total sum, stores them in a vector, and sorts the coins

 * in descending order. Then, it iteratively sums the largest coins while simultaneously subtracting

 * from the total sum of remaining coins until the sum of taken coins strictly exceeds the sum of remaining coins.

 * The count of these taken coins is then printed.

 * @algorithm Greedy approach with sorting a vector.

 * Time Complexity: $O(N \log N)$ due to sorting, where N is the number of coins.

 * Space Complexity: $O(N)$ to store the coin values in a vector.

 */



#include <algorithm> // Required for std::sort and std::greater.

#include <iostream>  // Required for std::cin and std::cout.

#include <vector>    // Required for std::vector.



using namespace std; // Uses the standard namespace.



int main() {

  /**

   * @brief Main function: orchestrates the coin selection logic.

   */

  int n; // @var n: The total number of coins.

  vector<int> coins; // @var coins: A dynamic array (vector) to store the coin values.



  cin >> n; // Reads the total number of coins.



  int total = 0; // @var total: Stores the total sum of all coin values.

  int val;       // @var val: Temporary variable to store each coin value as it's read.



  /**

   * @brief Reads coin values, stores them in `coins` vector, and calculates their total sum.

   * @block_logic Iterates `n` times to read each coin value and add it to `coins` and `total`.

   * @invariant `coins` vector is populated with input values, and `total` accumulates their sum.

   */

  for (int i = 0; i < n; ++i) {

    cin >> val;               // Reads a coin value.

    coins.push_back(val);     // Adds the coin value to the vector.

    total += val;             // Adds the coin value to the total sum.

  }



  /**

   * @brief Sorts the coin values in descending order.

   * @details This is a crucial step for the greedy approach, ensuring that the largest coins are considered first.

   * @param coins.begin() Iterator to the beginning of the vector.

   * @param coins.end()   Iterator to the end of the vector.

   * @param greater<int>() Custom comparator to sort in descending order.

   */

  sort(coins.begin(), coins.end(), greater<int>());



  int coin_count = 0; // @var coin_count: Counter for the number of coins taken.

  int share = 0;      // @var share: Accumulates the sum of coins taken.



  /**

   * @brief Greedily selects coins starting from the largest until their sum exceeds the sum of remaining coins.

   * @block_logic Iterates through the sorted coins, adding to `share` and subtracting from `total` until `share > total`.

   * @invariant `share` is the sum of `coin_count` largest coins, and `total` is the sum of remaining coins.

   */

  for (int i : coins) { // Range-based for loop for iterating through coins.

    coin_count++;        // Increments the count of selected coins.

    total -= i;          // Subtracts the current coin from the sum of remaining coins.

    share += i;          // Adds the current coin to the sum of selected coins.



    /**

     * @brief Checks if the sum of selected coins (`share`) is strictly greater than the sum of remaining coins (`total`).

     * @block_logic If the condition is met, prints the `coin_count` and terminates.

     * @invariant The loop breaks as soon as the greedy condition is satisfied.

     */

    if (share > total) {

      cout << coin_count; // Prints the minimum number of coins required.

      break;              // Exits the loop and the program.

    }

  }



  return 0; // Indicates successful execution (if loop completes without finding solution, though problem guarantees one).

}
