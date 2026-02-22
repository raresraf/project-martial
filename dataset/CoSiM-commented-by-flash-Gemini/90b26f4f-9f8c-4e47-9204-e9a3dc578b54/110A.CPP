/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number.
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if the count of its digits '4' and '7' is itself
 * a lucky number (i.e., exactly 4 or exactly 7). The program extracts digits, counts
 * occurrences of 4s and 7s, and then checks if this count is 4 or 7.
 * @algorithm Digit extraction and counting.
 * Time Complexity: $O(\log_{10} N)$ where N is the input number, as it iterates through its digits.
 * Space Complexity: $O(\log_{10} N)$ to store digits in an array, or $O(1)$ if counting directly.
 */
#include <bits/stdc++.h>
/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number.
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if the count of its digits '4' and '7' is itself
 * a lucky number (i.e., exactly 4 or exactly 7). The program extracts digits, counts
 * occurrences of 4s and 7s, and then checks if this count is 4 or 7.
 * @algorithm Digit extraction and counting.
 * Time Complexity: $O(\log_{10} N)$ where N is the input number, as it iterates through its digits.
 * Space Complexity: $O(\log_{10} N)$ to store digits in an array, or $O(1)$ if counting directly.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream, string, algorithm).
using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  long long n; // @var n: The input number to check.
  cin >> n; // Reads the input number 'n'.

  int j = 0;   // @var j: Index for populating the `arr` from right to left (least significant digit to most significant).
  int arr[100]; // @var arr: Array to store individual digits of `n`. Assumes `n` has at most 100 digits.

  // Functional Utility: Calculates the number of digits in `n`.
  int len = to_string(n).length(); // @var len: Stores the length (number of digits) of `n`.

  /**
   * @brief Extracts digits from `n` and stores them in `arr` in reverse order.
   * @block_logic Repeatedly takes the last digit of `n`, stores it, and removes it from `n`.
   * @invariant `n` is progressively reduced, and its digits are stored in `arr` from right to left.
   *            `j` counts the number of digits extracted.
   */
  while (n != 0) {
    arr[len - j - 1] = n % 10; // Stores the last digit of `n` into `arr` at the correct position.
    n = n / 10;                // Removes the last digit from `n`.
    j++;                       // Increments the count of extracted digits.
  }

  int i = 0;     // @var i: Index for iterating through the `arr` from left to right.
  int count = 0; // @var count: Counter for the occurrences of lucky digits (4 or 7).

  /**
   * @brief Iterates through the extracted digits to count occurrences of '4' or '7'.
   * @block_logic Processes each digit in `arr` to identify lucky digits.
   * @invariant `len` decreases, `i` increases, and `count` accumulates occurrences of '4' or '7'.
   */
  while (len) {
    if (arr[i] == 4 || arr[i] == 7) // Checks if the current digit is 4 or 7.
      count++;                     // Increments the lucky digit counter.
    len--;                         // Decrements the remaining length.
    i++;                           // Moves to the next digit.
  }

  /**
   * @brief Determines if the count of lucky digits is itself a lucky number (4 or 7).
   * @block_logic Checks the final `count` value against the lucky numbers 4 and 7.
   * @invariant Prints "YES" if `count` is 4 or 7, otherwise "NO".
   */
  if (count == 4 || count == 7)
    cout << "YES"; // Output "YES" if the count of lucky digits is lucky.
  else
    cout << "NO";  // Output "NO" otherwise.

  return 0; // Indicates successful execution.
}
