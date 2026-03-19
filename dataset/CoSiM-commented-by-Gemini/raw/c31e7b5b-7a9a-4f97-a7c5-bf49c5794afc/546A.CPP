/**
 * @file 546A.CPP
 * @brief Compact C-style solution for the "Soldier and Bananas" problem (Codeforces 546A).
 * @details This program calculates the amount of money a soldier needs to borrow
 * to buy 'w' bananas, where the cost increases arithmetically.
 *
 * Algorithm: One-Line Arithmetic Series Calculation
 * The total cost of the bananas is an arithmetic series `k * (1 + 2 + ... + w)`.
 * This is calculated using the formula `k * (w * (w + 1) / 2)`. The program
 * computes the total cost, subtracts the money the soldier has (`n`), and if
 * the result is negative (meaning no borrowing is needed), it clamps the value to 0.
 *
 * NOTE: The implementation uses C-style I/O (`scanf`, `printf`) with incorrect
 * format specifiers (`%lu` for `long` types instead of `%ld`), which can lead
 * to undefined behavior on some platforms.
 *
 * Time Complexity: O(1), as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1).
 */

#include <stdio.h>
using namespace std;

int main() {
  // k: base cost of a banana.
  // w: number of bananas to buy.
  int k, w;
  // n: initial money the soldier has.
  long n;
  
  // Read inputs using scanf.
  // WARNING: Using "%lu" for a 'long n' is incorrect and may cause issues.
  // The correct specifier is "%ld".
  scanf("%d %lu %d", &k, &n, &w);
  
  // Core Logic (in one line):
  // 1. (w * (w + 1) / 2) calculates the sum of the first 'w' integers.
  // 2. k * (...) calculates the total cost.
  // 3. - n calculates the deficit.
  long r = (k * (w * (w + 1) / 2) - n);
  
  // If the result 'r' is negative (soldier has enough money), set it to 0.
  // Otherwise, keep the positive deficit value.
  r = r > 0 ? r : 0;
  
  // Print the result.
  // WARNING: Using "%lu" for a 'long r' is incorrect.
  printf("%lu", r);
  
  return 0;
}