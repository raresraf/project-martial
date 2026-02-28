/**
 * @file 339A.CPP
 * @brief Sorts digits within a string representing a sum (e.g., "3+1+2" to "1+2+3").
 *
 * This program reads a string which is expected to be a sum of single-digit
 * numbers separated by '+' characters (e.g., "3+1+2"). The goal is to sort
 * these digits in non-decreasing order while preserving the '+' separators,
 * and then print the resulting sorted sum string.
 *
 * Algorithm (Intended): The `solve` function attempts to sort the digits
 *                        (characters at even indices) using a selection sort
 *                        or bubble sort-like approach in place.
 * Note: The current implementation compares characters lexicographically (e.g., '1' < '2'),
 *       and the swap operation might be problematic if not handled carefully with `char` types.
 *
 * Time Complexity: O(L^2) where L is the length of the input string. This is
 *                  due to the nested loops iterating over approximately L/2 elements.
 * Space Complexity: O(1) as the sorting is performed in-place on the input string.
 */

#include <algorithm> // Provides std::swap (though not explicitly used as `std::swap` in the original code).
#include <iostream>
using namespace std;

// This macro definition is syntactically incorrect (missing semicolon, 'long long' cannot be 'll' without type).
// It appears to be an attempt to define a type alias `ll` for `long long`.
#define long long ll;

string s; // Global variable to store the input sum string.

void solve() {
  string answer; // Local variable 'answer' is declared but not used.
  /**
   * @brief Block Logic: Iterates through the string to sort the digits.
   *
   * This outer loop processes each digit position (even indices). For each digit,
   * it finds the smallest digit among the remaining unsorted digits and effectively
   * moves it to the current `i` position.
   * Invariant: After `i` iterations, the digits at indices `0, 2, ..., i-2` are sorted.
   */
  for (int i = 0; i < s.length(); i = i + 2) {
    /**
     * @brief Block Logic: Inner loop to find the minimum digit and perform a swap.
     *
     * It compares the digit at `s[i]` with subsequent digits at even indices `s[j]`.
     * If `s[i]` is greater than `s[j]`, they are swapped. This effectively bubbles
     * the smaller digit towards the left for each pass of the outer loop.
     * Invariant: After `j` iterations in the inner loop, `s[i]` holds the minimum
     *            digit found so far among `s[i], s[i+2], ..., s[j]`.
     */
    for (int j = i; j < s.length(); j = j + 2) {
      // @brief Compares characters lexicographically.
      // Note: This directly compares character codes, which works for digits '0'-'9'
      //       to compare their numeric values correctly.
      if (s[i] > s[j]) {
        // @brief Performs a manual swap of characters.
        // Note: The use of `int aux = s[i];` is correct as characters can be
        //       implicitly converted to int for temporary storage.
        int aux = s[i];
        s[i] = s[j];
        s[j] = aux;
      }
    }
    // @brief Conditional printing of the digit and '+' separator.
    // Logic: Prints '+' after a digit unless it is the last digit in the string.
    if (i != s.length() - 1)
      cout << s[i] << "+";
    else
      cout << s[i];
  }
}

int main() {
  /**
   * @brief Main function to read input and invoke the sorting logic.
   */
  cin >> s; // Reads the input sum string.

  solve(); // Calls the function to sort and print the digits.
}