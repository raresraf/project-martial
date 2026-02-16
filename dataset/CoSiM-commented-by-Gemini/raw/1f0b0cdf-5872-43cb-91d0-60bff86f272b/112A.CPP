/**
 * @file 112A.CPP
 * @brief Solution for Codeforces problem 112A, "Petya and Strings".
 *
 * @description
 * This program performs a case-insensitive lexicographical comparison of two
 * input strings. It prints -1 if the first string is lexicographically smaller
 * than the second, 1 if it is larger, and 0 if they are equal.
 *
 * This implementation is robust and idiomatic, using the standard C++ `<locale>`
 * library for case conversion and separating the character comparison logic
 * into a helper function.
 *
 * @algorithm
 * 1. A helper function `compare(char, char)` is defined to perform a
 *    case-insensitive comparison of two characters using `std::tolower`.
 * 2. The `main` function reads two strings.
 * 3. It iterates through the strings character by character, from left to right.
 * 4. In each iteration, it calls the `compare` function on the current pair of
 *    characters.
 * 5. If the `compare` function returns a non-zero value (-1 or 1), a difference
 *    has been found. The result is recorded, and the loop is terminated immediately.
 * 6. If the characters are equal, the loop continues.
 * 7. If the loop completes without finding any differences, the strings are equal,
 *    and the result remains 0.
 * 8. The final result is printed.
 *
 * @time_complexity O(L), where L is the length of the strings. In the best
 *   case, it's O(1) if the strings differ at the first character.
 * @space_complexity O(L) to store the two strings.
 */
#include <iostream> // std::cout
#include <locale>   // std::locale, std::tolower
#include <string>   // std::string

/**
 * @brief Performs a case-insensitive comparison of two characters.
 * @param a The first character.
 * @param b The second character.
 * @return -1 if a < b, 1 if a > b, 0 if a == b (all case-insensitive).
 */
int compare(char a, char b) {
  std::locale loc;
  // Use the standard, locale-aware tolower function for robust case conversion.
  char lowerA = std::tolower(a, loc);
  char lowerB = std::tolower(b, loc);

  if (lowerA < lowerB)
    return -1;
  else if (lowerA > lowerB)
    return 1;
  else
    return 0;
}

int main() {
  std::string firstString, secondString;
  int stringLength;
  // `answer` stores the result of the comparison, defaulting to 0 for equality.
  int answer = 0;

  std::cin >> firstString >> secondString;

  stringLength = firstString.length();

  // Block Logic: Iterate through the strings and compare them character by character.
  for (int i = 0; i < stringLength; i++) {
    int temp = compare(firstString[i], secondString[i]);

    // If a difference is found, set the answer and exit the loop immediately.
    if (temp == -1) {
      answer = -1;
      break;
    } else if (temp == 1) {
      answer = 1;
      break;
    }
  }

  std::cout << answer << std::endl;

  return 0;
}
