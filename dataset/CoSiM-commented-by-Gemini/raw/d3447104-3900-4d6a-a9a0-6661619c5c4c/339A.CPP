/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem (Codeforces 339A) using a general sorting approach.
 * @details This program reads a string representing a sum (e.g., "3+2+1") and
 * reorders it into a non-decreasing sequence (e.g., "1+2+3").
 *
 * @section algorithm
 * Algorithm: Standard Sort
 * 1. The input string containing the sum is read.
 * 2. The string is parsed: digit characters are converted to integers and stored
 *    in an auxiliary array, while '+' characters are ignored.
 * 3. The C++ standard library's `std::sort` function is called on the array to
 *    sort the numbers in non-decreasing order.
 * 4. The program then iterates through the sorted array, printing each number
 *    and interleaving '+' symbols to reconstruct the sorted sum string.
 *
 * @section complexity
 * - Time Complexity: O(N log N), where N is the number of terms in the sum.
 *   This is dominated by the `std::sort` algorithm. The parsing and printing
 *   steps are linear.
 * - Space Complexity: O(N), to store the numbers in the auxiliary array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int arr[1001];
  string s;
  cin >> s;
  int count = 0;

  /**
   * @brief This loop parses the input string.
   * It extracts the digit characters, converts them to integers, and stores
   * them in the `arr` array. The `+` symbols are skipped.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '+') {
      continue;
    } else {
      arr[count++] = s[i] - '0';
    }
  }

  /**
   * @brief Sorts the array of numbers.
   * `std::sort` is a highly optimized, typically introsort-based algorithm.
   */
  sort(arr, arr + count);

  /**
   * @brief Reconstructs and prints the sorted sum.
   * This loop iterates through the sorted numbers and prints them, separated
   * by '+' signs.
   */
  for (int i = 0; i < count; i++) {
    cout << arr[i];
    if (i == count - 1) {
      break; // Avoids printing a '+' after the last number.
    }
    cout << "+";
  }
  return 0;
}