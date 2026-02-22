/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A: Petya and Strings.
 * @details This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them. The output indicates their relative
 * lexicographical order: -1 if the first string is smaller, 1 if the first string is greater,
 * and 0 if they are equal.
 * @algorithm Case-insensitive lexicographical string comparison.
 * Time Complexity: $O(L)$ where L is the length of the strings. The `tolower` operation and string comparison both take linear time.
 * Space Complexity: $O(L)$ to store the input strings, or $O(1)$ if modifications are in-place and not new strings.
 */
#include <bits/stdc++.h>

using namespace std;

/**

 * @file 112A.CPP

 * @brief Solution for Codeforces Problem 112A: Petya and Strings.

 * @details This program performs a case-insensitive lexicographical comparison of two input strings.

 * It converts both strings to lowercase and then compares them. The output indicates their relative

 * lexicographical order: -1 if the first string is smaller, 1 if the first string is greater,

 * and 0 if they are equal.

 * @algorithm Case-insensitive lexicographical string comparison.

 * Time Complexity: $O(L)$ where L is the length of the strings. The `tolower` operation and string comparison both take linear time.

 * Space Complexity: $O(L)$ to store the input strings, or $O(1)$ if modifications are in-place and not new strings.

 */

#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream, string, algorithm, cctype).



using namespace std; // Using the standard namespace to avoid prefixing std::.



int main() {

  /**

   * @brief Main function: orchestrates the string comparison logic.

   */



  string s1, s2; // @var s1, s2: The two input strings to be compared.



  cin >> s1; // Reads the first string from standard input.

  cin >> s2; // Reads the second string from standard input.



  /**

   * @brief Converts both strings to lowercase for case-insensitive comparison.

   * @block_logic Iterates through each character of both strings and converts them to lowercase.

   * @invariant After the loop, `s1` and `s2` contain their lowercase equivalents.

   */

  for (int i = 0; i < s1.length(); ++i) {

    s1[i] = tolower(s1[i]); // Converts character at index `i` in `s1` to lowercase.

    s2[i] = tolower(s2[i]); // Converts character at index `i` in `s2` to lowercase.

  }



  /**

   * @brief Performs lexicographical comparison of the lowercase strings and prints the result.

   * @block_logic Uses standard string comparison operators to determine relative order.

   * @invariant Prints "1", "0", or "-1" based on the comparison result.

   */

  if (s1 > s2)

    printf("1\n");        // `s1` is lexicographically greater than `s2`.

  else if (s1 == s2)

    printf("0\n");       // `s1` is lexicographically equal to `s2`.

  else

    printf("-1\n");      // `s1` is lexicographically smaller than `s2`.



  return 0; // Indicates successful execution.

}
