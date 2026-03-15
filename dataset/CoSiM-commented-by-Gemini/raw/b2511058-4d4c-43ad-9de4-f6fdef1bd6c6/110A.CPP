/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem with an interesting logical flaw.
 * @details A number is "nearly lucky" if the count of its lucky digits ('4' and '7')
 * is itself a lucky number (4 or 7). This program correctly counts the lucky digits
 * but then uses a flawed-but-functional method to check if the count is lucky.
 *
 * @algorithm
 * 1. The input number is read as a string.
 * 2. The code iterates through the string, counting the occurrences of '4' and '7'.
 * 3. This integer count is then converted back into a string, `luckstr`.
 * 4. The code then checks if `luckstr` is composed entirely of the digits '4' and '7'.
 *    - If it finds any digit in `luckstr` that is not '4' or '7', it prints "NO".
 *    - If all digits in `luckstr` are '4' or '7', it prints "YES".
 *
 * @note This implementation has a logical error. The problem requires checking if the
 * count IS 4 or 7. This code checks if the count is COMPOSED OF 4s and 7s (e.g., it
 * would incorrectly pass for a count of 47). However, since the input number has at most
 * 18 digits, the maximum possible count of lucky digits is 18. The only numbers
 * in the range [1, 18] that are composed of only '4's and '7's are 4 and 7.
 * Thus, this flawed logic coincidentally produces the correct answer for the problem's constraints.
 *
 * @complexity
 * - Time: O(L), where L is the number of digits in the input number.
 * - Space: O(L) for the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string n; // Stores the input number as a string.
  cin >> n;
  
  // `lucky` will store the count of lucky digits found in the input string.
  int lucky = 0;

  /**
   * @brief This loop counts the occurrences of lucky digits '4' and '7'.
   */
  for (int i = 0; i < n.length(); i++) {
    if (n[i] == '4' || n[i] == '7') {
      lucky++;
    }
  }

  // Convert the integer count back to a string for digit-by-digit inspection.
  string luckstr = to_string(lucky);

  /**
   * @brief This loop checks if the `lucky` count itself is composed of lucky digits.
   * This is a flawed interpretation of the problem, but it works due to the problem's constraints.
   */
  for (int i = 0; i < luckstr.length(); i++) {
    if (!(luckstr[i] == '4' || luckstr[i] == '7')) {
      cout << "NO" << endl;
      return 0; // Exit if a non-lucky digit is found in the count.
    }
  }
  
  // If the loop completes, it means the count was composed only of '4's and/or '7's.
  cout << "YES" << endl;
}