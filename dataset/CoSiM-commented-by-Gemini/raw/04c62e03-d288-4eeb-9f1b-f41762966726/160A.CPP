/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces contest 160.
 *
 * @details
 * The problem asks for the minimum number of coins one needs to take from a set of coins
 * such that the sum of the values of the taken coins is strictly greater than the sum of the
 * values of the remaining coins.
 *
 * To solve this optimally, one should always take the coins with the highest value first. This is a
 * greedy approach. The program first reads all the coin values and calculates their total sum.
 * Then, it sorts the coins in descending order. It iterates through the sorted coins, adding
 * their values to a running sum for the coins taken. The process stops as soon as the sum of
 * the taken coins exceeds half of the total sum. The number of coins taken up to that point
 * is the minimum required.
 *
 * Algorithm:
 * 1. Read the number of coins, `n`.
 * 2. Read `n` coin values into a vector and calculate the total sum of all coins.
 * 3. Sort the vector of coin values in descending (non-increasing) order.
 * 4. Initialize a `taken_sum` and a `coin_count` to zero.
 * 5. Iterate through the sorted coins:
 *    a. Add the value of the current coin to `taken_sum`.
 *    b. Increment `coin_count`.
 *    c. Check if `taken_sum` is now greater than `total_sum / 2`. If it is, stop.
 * 6. Print `coin_count`.
 *
 * Time Complexity: O(n log n), dominated by the sorting operation.
 * Space Complexity: O(n) to store the coin values in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, sum = 0, suma = 0, c = 0;
  // Read the number of coins.
  std::cin >> n;
  int k;
  std::vector<int> x;
  
  // Block Logic: Read all coin values, store them in a vector, and calculate the total sum.
  for (int i = 0; i < n; i++) {
    std::cin >> k;
    x.push_back(k);
    sum += k;
  }
  
  // Functional Utility: Sort the coins in descending order to apply a greedy strategy.
  std::sort(x.rbegin(), x.rend());
  
  // Block Logic: Greedily take the largest coins until their sum is strictly greater than the sum of the rest.
  // The condition `suma <= sum / 2` is equivalent to checking if our sum is not yet greater than the other half.
  // Invariant: 'suma' holds the sum of the 'c' largest coins.
  for (int j = 0; suma <= sum / 2; j++) {
    suma += x[j];
    c++;
  }
  
  // Output the minimum number of coins required.
  std::cout << c;
  return 0;
}