/**
 * @b405b98a-3133-416d-ba0f-97f92189270d/160A.CPP
 * @brief Solution for a competitive programming problem to find the minimum coins for a sum greater than half the total.
 *
 * This program efficiently determines the minimum number of coins whose sum is strictly
 * greater than half of the total sum of all available coins. It uses a greedy approach
 * by taking the largest coins first, leveraging C++ standard library features like vectors and sorting.
 *
 * Algorithm:
 * 1. Read the total number of coins 'a'.
 * 2. Read 'a' coin values into a dynamic array (std::vector) and compute their total sum.
 * 3. Calculate the threshold: half of the total sum.
 * 4. Sort the coins in non-decreasing order.
 * 5. Iterate from the largest coin, accumulating their values and counting them,
 *    until the accumulated sum strictly exceeds the calculated threshold.
 * 6. Print the count of coins collected.
 *
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values in a vector.
 */

#include <algorithm> // Functional Utility: Provides the std::sort algorithm for sorting elements.
#include <iostream>  // Functional Utility: Provides input/output functionalities (std::cin, std::cout).
#include <vector>    // Functional Utility: Provides the std::vector container for dynamic arrays.

// Functional Utility: Macro to optimize C++ standard I/O operations by decoupling C++ streams
// from standard C streams and untying cin from cout. Essential for competitive programming speed.
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  // Functional Utility: Applies I/O optimizations for faster execution.
  optimizar_io
  long long a, sum = 0, sum1 = 0, cont = 0;
  // Functional Utility: Reads the total number of coins 'a'.
  cin >> a;
  // Functional Utility: Declares a vector of long long integers of size 'a' to store coin values.
  vector<long long> v(a);
  // Block Logic: Reads coin values into the vector and calculates their total sum.
  // Pre-condition: 'a' integer values representing coin values are available in input.
  // Invariant: 'v' will contain all coin values, and 'sum' will hold their total sum.
  for (int i = 0; i < a; ++i) {
    cin >> v[i];
    sum += v[i];
  }
  // Functional Utility: Calculates the target sum (half of the total sum) for comparison.
  // Integer division here implies we need to strictly exceed this value.
  sum /= 2;
  // Functional Utility: Sorts the vector of coin values in non-decreasing order.
  sort(v.begin(), v.end());
  // Block Logic: Greedily selects coins from largest to smallest until their sum exceeds the target.
  // Pre-condition: 'v' is sorted; 'sum' is the target half-sum.
  // Invariant: 'sum1' accumulates values, 'cont' counts coins, loop breaks when 'sum1' > 'sum'.
  for (int i = a - 1; i >= 0; --i) { // Functional Utility: Iterates from the largest coin.
    sum1 += v[i]; // Functional Utility: Adds the current coin's value to the accumulated sum.
    cont++;       // Functional Utility: Increments the count of coins taken.
    // Conditional Logic: Checks if the accumulated sum now exceeds the target half-sum.
    if (sum1 > sum)
      break; // Functional Utility: Exits the loop as the condition is met.
  }
  // Functional Utility: Prints the minimum number of coins required, followed by a newline.
  cout << cont << "\n";
  return 0; // Functional Utility: Indicates successful program execution.
}
