/**
 * @file 339A.CPP
 * @brief Sorts the terms in a mathematical sum represented as a string.
 *
 * The input is a string representing a sum of single-digit numbers,
 * e.g., "3+2+1". The program outputs the sum with the terms rearranged in
 * non-decreasing order, e.g., "1+2+3".
 *
 * @algorithm
 * 1. Extract all digits from the input string and store them in a character vector.
 * 2. Use `std::sort` to sort the vector of digits.
 * 3. Reconstruct the sum by printing the sorted digits, interleaved with '+' signs.
 *
 * @time_complexity O(N log N), where N is the number of digits in the sum. The
 *                  dominant operation is the sorting of the vector.
 * @space_complexity O(N) to store the digits in the vector.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main(void) {
  string test;
  cin >> test;
  vector<char> vec;

  /**
   * @block_logic Extracts the numbers from the sum string.
   * It iterates with a step of 2 to skip the '+' characters.
   */
  for (int i = 0; i < test.length(); i += 2) {
    vec.push_back(test[i]);
  }

  // Sorts the vector of number characters.
  sort(vec.begin(), vec.end());

  /**
   * @block_logic Reconstructs and prints the sorted sum.
   * It prints a '+' after each number except for the last one.
   */
  for (int i = 0; i < vec.size(); i++) {
    cout << vec[i];

    if (i != (vec.size() - 1)) {
      cout << "+";
    }
  }
  return 0;
}