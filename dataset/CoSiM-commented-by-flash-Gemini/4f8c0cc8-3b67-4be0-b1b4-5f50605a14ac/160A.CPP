/**
 * @file 160A.CPP
 * @brief Determines the minimum number of largest coins required such that their sum is strictly greater than the sum of the remaining coins.
 *
 * Problem Name: Twin Coins / Helpful Maths (Codeforces 160A)
 * Algorithm: Greedy approach. It sorts coins in descending order and iteratively picks the largest ones
 *            until the sum of chosen coins exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values in a `std::vector`.
 */
#include <algorithm> // Required for std::sort.
#include <cstdio>    // Required for getchar and potentially printf (though iostream used for output).
#include <cstdlib>   // Not directly used.
#include <iostream>  // Required for std::cout.
#include <vector>    // Required for std::vector.
using namespace std;
#define gc getchar_unlocked // Functional Utility: Macro for fast character input.

/**
 * @brief Custom fast input function to read an integer.
 * @param x Reference to an integer variable where the scanned value will be stored.
 * Functional Utility: Optimizes input reading for large datasets in competitive programming.
 */
inline int getchar_unlocked() {
  return getchar();
} // Fallback to getchar if getchar_unlocked is not available or desired.

void scanint(int &x) {
  register int c = gc(); // Reads characters one by one.
  x = 0;
  int neg = 0; // Flag for negative numbers.
  // Block Logic: Skips non-digit characters and handles optional negative sign.
  for (; ((c < 48 || c > 57) && c != '-'); c = gc())
    ;
  if (c == '-') {
    neg = 1;
    c = gc();
  }
  // Block Logic: Accumulates digits to form the integer value.
  for (; c > 47 && c < 58; c = gc()) {
    x = (x << 1) + (x << 3) + c - 48; // Efficient way to calculate x = x*10 + digit.
  }
  if (neg)
    x = -x; // Applies negative sign if present.
}

/**
 * @brief Main function to solve the twin coins problem.
 * @pre Input consists of an integer 'n' (number of coins), followed by 'n' integers representing coin values.
 * @post Prints the minimum number of coins whose sum is strictly greater than the sum of the rest.
 */
int main() {
  int n, i, j, sum = 0, taken = 0; // 'n': number of coins, 'sum': total sum of all coins (then remaining sum),
                                   // 'taken': sum of coins chosen, 'i,j': loop counters.
  ; // Empty statement, often a typo or placeholder in competitive programming.
  vector<int> a; // Vector to store coin values.

  // Functional Utility: Reads the number of coins using custom fast input.
  scanint(n);

  /**
   * Block Logic: Reads coin values, calculates their total sum, and stores them in a vector.
   * Invariant: `sum` accumulates the total value, and `a` stores each coin value.
   */
  for (i = 0; i < n; i++) {
    scanint(j); // Reads individual coin value.
    sum += j;   // Adds to total sum.
    a.push_back(j); // Stores in vector.
  }

  // Functional Utility: Sorts the coins in descending order. `rbegin()` and `rend()` provide reverse iterators.
  sort(a.rbegin(), a.rend());

  // Block Logic: Comments for debugging purposes, not executable code.
  // for(i=0; i<n; i++)
  //	cout<<a[i]<<endl;

  i = 0; // Resets 'i' for the next loop, acting as a counter for coins taken.

  /**
   * Block Logic: Greedily picks coins from largest to smallest until 'taken' sum exceeds 'sum' of remaining coins.
   * Invariant: `taken` increases, `sum` (representing remaining coins) decreases. The loop counts how many coins are picked.
   */
  while (taken <= sum) {
    taken += a[i]; // Adds the current largest coin to the 'taken' sum.
    sum = sum - a[i]; // Subtracts it from the total sum (which now represents remaining).
    i++;             // Increments count of coins taken.
  }

  // Block Logic: Comments for debugging purposes, not executable code.
  // cout<<"sum "<<sum<<"taken "<<taken<<" "<<i<<endl;

  // Functional Utility: Prints the minimum number of coins required.
  cout << i << endl; // 'i' holds the count of coins taken.
  // Indicates successful program execution.
  return 0;
}