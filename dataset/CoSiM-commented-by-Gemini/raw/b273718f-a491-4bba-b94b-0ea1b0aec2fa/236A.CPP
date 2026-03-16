/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - "Boy or Girl".
 * @details This program determines whether the number of distinct characters in a given
 *          username string is even or odd. Based on this, it prints one of two messages.
 *
 * Algorithm:
 * 1. Read the input username string.
 * 2. Count the number of distinct characters in the string.
 *    - The implementation uses a naive O(N^2) approach. It iterates through the string,
 *      and for each character, it checks if a duplicate exists later in the string.
 *      It counts how many characters have such duplicates.
 *    - The number of distinct characters is then calculated by subtracting this count
 *      from the total length of the string.
 * 3. Check if the resulting count of distinct characters is even or odd.
 * 4. If the count is even, print "CHAT WITH HER!".
 * 5. If the count is odd, print "IGNORE HIM!".
 *
 * Time Complexity: O(N^2), where N is the length of the string, due to the nested loops for duplicate checking.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  int len, i, j, n = 0, f;
  string str;

  cin >> str;
  len = str.size();

  /**
   * @brief Block Logic: Count characters that are not the last occurrence of their kind.
   * This is a naive O(N^2) method to ultimately find the number of unique characters.
   * 'n' accumulates a count of characters for which a duplicate exists at a later index.
   */
  for (i = 0; i < len; i++) {
    f = 0;
    // Inner loop to find if a duplicate of str[i] exists after it.
    for (j = i + 1; j < len; j++) {
      if (str[i] == str[j]) {
        // A duplicate is found.
        f = 1;
      }
    }
    // If a duplicate was found later in the string, increment the non-unique count.
    if (f == 1)
      n++;
  }

  // Calculate the number of distinct characters.
  len -= n;

  /**
   * @brief Block Logic: Determine the output based on the parity of the distinct character count.
   * Invariant: `len` now holds the number of distinct characters in the input string.
   */
  if (len % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
