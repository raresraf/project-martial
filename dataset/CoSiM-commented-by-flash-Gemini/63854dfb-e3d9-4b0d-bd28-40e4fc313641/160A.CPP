/**
 * @63854dfb-e3d9-4b0d-bd28-40e4fc313641/160A.CPP
 * @brief Solution for a competitive programming problem to find the minimum coins for a sum greater than half the total.
 *
 * This program determines the minimum number of coins required to achieve a sum
 * that is strictly greater than half of the total sum of all available coins.
 * It uses a greedy strategy by always picking the largest available coins first.
 * This particular implementation uses a vector and repeatedly removes the largest
 * element after sorting and reversing.
 *
 * Algorithm:
 * 1. Read the number of coins 'n'.
 * 2. Read the values of all 'n' coins, store them in a vector, and calculate their total sum.
 * 3. Sort the coins in ascending order, then reverse the vector to get them in descending order.
 * 4. Initialize 'mySum' (accumulated sum of taken coins) to 0 and 'ans' (count of coins taken) to 0.
 * 5. Iterate while 'mySum' is less than or equal to half of the 'total' sum:
 *    a. Take the largest available coin (front of the reversed vector).
 *    b. Add its value to 'mySum'.
 *    c. Increment 'ans'.
 *    d. Remove the taken coin from the vector.
 * 6. Print the final count of 'ans'.
 *
 * Time Complexity: O(N log N) due to sorting, and O(N) for vector erase operations in the worst case,
 *                  making the overall complexity dominated by sorting.
 * Space Complexity: O(N) to store the coin values in a vector.
 */

#include <algorithm> // Functional Utility: Provides std::sort and std::reverse.
#include <iostream>  // Functional Utility: Provides std::cout (though scanf/printf are used).
#include <vector>    // Functional Utility: Provides std::vector for dynamic array management.
using namespace std;

int main(int argc, char *argv[]) { // Functional Utility: Standard main function signature.
  int n, ans = 0, total = 0, mySum = 0;
  vector<int> coins;
  // Functional Utility: Reads the number of coins 'n'.
  scanf("%d", &n);
  // Block Logic: Reads 'n' coin values, calculates their total sum, and stores them in a vector.
  // Pre-condition: 'n' integer coin values are available in input.
  // Invariant: 'total' holds the sum of all coins; 'coins' vector contains all coin values.
  for (int i = 0; i < n; i++) {
    int coin;
    scanf("%d", &coin);
    total += coin;
    coins.push_back(coin);
  }
  // Functional Utility: Sorts the coins in ascending order.
  sort(coins.begin(), coins.end());
  // Functional Utility: Reverses the sorted coins to arrange them in descending order.
  reverse(coins.begin(), coins.end());
  // Block Logic: Greedily takes coins (largest first) until 'mySum' exceeds half of 'total'.
  // Pre-condition: 'coins' vector is sorted in descending order; 'total' is the sum of all coins.
  // Invariant: 'mySum' accumulates coin values, 'ans' counts selected coins, loop terminates when 'mySum' > 'total' / 2.
  while (mySum <= float(total) / 2.) { // Inline: Uses float conversion for precise half-sum comparison.
    mySum += coins.front(); // Functional Utility: Adds the largest available coin to 'mySum'.
    ans++;                  // Functional Utility: Increments the count of selected coins.
    coins.erase(coins.begin()); // Functional Utility: Removes the largest coin from the vector.
  }
  // Functional Utility: Prints the final count of coins taken.
  printf("%d\n", ans);

  return 0; // Functional Utility: Indicates successful program execution.
}
