/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces.
 * @details The problem defines a "lucky number" as containing only digits 4 and 7.
 * A number is "nearly lucky" if the count of its lucky digits is itself a lucky number.
 *
 * @algorithm
 * This solution first counts the lucky digits and then uses a regular expression
 * to determine if the resulting count is a "lucky number".
 * 1. Read the input number and convert it to a string for easy iteration.
 * 2. Initialize a counter `count` to 0.
 * 3. Iterate through each character of the number string.
 *    a. Convert the character digit to its integer value.
 *    b. If the digit is 4 or 7, increment `count`.
 * 4. Convert the final `count` back into a string.
 * 5. Use `std::regex_match` to check if this new string is composed entirely
 *    of the digits '4' and '7'. The pattern `[47]*` matches a string with
 *    zero or more characters that are either '4' or '7'.
 * 6. Print "YES" if it matches, "NO" otherwise.
 *
 * @note This regex approach is a very general way to check if the count is a
 * lucky number. For this specific problem's constraints, a simpler check like
 * `if (count == 4 || count == 7)` would also work and be more efficient.
 *
 * @complexity
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the number as a string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  long long num;
  cin >> num;

  string str = to_string(num);
  int count = 0;
  
  /**
   * @block
   * @brief Count the number of lucky digits (4 or 7) in the input number.
   */
  for (char c : str) {
    // Convert character to integer value (e.g., '4' -> 4).
    int l = (int)c - 48;

    if (l == 4 || l == 7)
      count++;
  }

  // Define a regex pattern for a "lucky number": contains only 4s and 7s.
  regex pattern("[47]*");
  // Convert the final count to a string to be checked against the pattern.
  string nl = to_string(count);

  /**
   * @brief Check if the count is a "lucky number" using regex.
   * `regex_match` checks if the entire string `nl` matches the pattern.
   */
  regex_match(nl, pattern) ? cout << "YES" : cout << "NO";
  cout << endl;

  return 0;
}
