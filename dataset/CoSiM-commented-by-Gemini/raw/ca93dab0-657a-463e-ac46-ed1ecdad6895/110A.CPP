/**
 * @file 110A.CPP
 * @brief Solution for Codeforces problem 110A, "Nearly Lucky Number".
 * @details This program checks if a given number is "nearly lucky". A number is defined
 * as nearly lucky if the total count of its "lucky" digits (which are '4' and '7')
 * is itself a lucky number (i.e., the count is 4 or 7).
 *
 * @section Algorithm
 * The algorithm reads the input number as a string to facilitate digit-by-digit
 * processing.
 * 1. It iterates through each character of the input string.
 * 2. It maintains a counter, `c`, which is incremented every time a '4' or a '7'
 *    is encountered.
 * 3. After counting all the lucky digits, it checks if the final count `c` is
 *    equal to 4 or 7.
 * 4. If the count is 4 or 7, it prints "YES"; otherwise, it prints "NO".
 *
 * Note: The `main` function contains a redundant, non-functional declaration
 * `void luck_check();` which has no effect on the program's execution.
 *
 * @subsection Complexity
 * - Time Complexity: O(L), where L is the number of digits in the input number,
 *   as the program performs a single pass over the string.
 * - Space Complexity: O(L), to store the input number as a string.
 */
#include <iostream>
using namespace std;

void luck_check(string x);

int main() {
  string num;
  cin >> num;
  luck_check(num);
  // The following line is a redundant function declaration and serves no purpose.
  void luck_check();
  return 0;
}

/**
 * @brief Counts the lucky digits ('4' and '7') in a string and checks if the
 * count itself is a lucky number (4 or 7).
 * @param x The input number represented as a string.
 */
void luck_check(string x) {
  // `c` counts the number of lucky digits found.
  unsigned long c = 0;
  /**
   * @brief This loop iterates through the string to count occurrences of '4' and '7'.
   */
  for (unsigned long i = 0; i < x.size(); i++) {
    if (x[i] == '7' || x[i] == '4') {
      c++;
    }
  }

  /**
   * @brief This block checks if the total count of lucky digits is 4 or 7.
   */
  if (c == 4 || c == 7) {
    cout << "YES";
  } else
    cout << "NO";
}
