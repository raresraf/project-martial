/**
 * @file 236A.CPP
 * @brief This program determines a response based on the parity of the number of distinct characters in an input string.
 * The input string is first sorted, and then the unique characters are counted.
 * If the count of distinct characters is odd, it prints "IGNORE HIM!"; otherwise, it prints "CHAT WITH HER!".
 * This problem is a typical competitive programming exercise testing basic string manipulation and conditional logic.
 * Algorithm: Sorting followed by a linear scan to count distinct elements.
 * Time Complexity: O(N log N) due to the sorting step, where N is the length of the input string.
 * Space Complexity: O(N) for storing the string (or O(1) if sorting in-place on a char array is considered constant auxiliary space).
 */
// C++ program to sort a string of characters
#include <bits/stdc++.h> // Includes most standard C++ libraries, including string manipulation and algorithm for sorting.
using namespace std;

int main() {
  char s[101]; // Declares a character array of size 101 to store the input string.
  int i, n, counter = 0; // Declares loop variable 'i', string length 'n', and a counter for distinct characters.
  cin >> s; // Reads the input string into the character array 's'.
  n = strlen(s); // Calculates the length of the string 's'.
  /**
   * Functional Utility: Sorts the characters in the string 's' in ascending order.
   * This simplifies counting distinct characters as identical characters will be adjacent.
   */
  sort(s, s + n);

  /**
   * Block Logic: Iterates through the sorted string to count the number of distinct characters.
   * It increments 'counter' only when a character is different from the next one,
   * effectively counting transitions between unique characters.
   */
  for (i = 0; i < n; i++) {
    // Conditional Logic: Checks if the current character is different from the next character.
    // This condition is true for the last occurrence of each distinct character.
    if (s[i] != s[i + 1])
      counter++; // Increments the counter for distinct characters.
  }

  /**
   * Conditional Logic: Determines the output based on the parity of the distinct character count.
   * If 'counter' is odd, it prints "IGNORE HIM!".
   * If 'counter' is even, it prints "CHAT WITH HER!".
   */
  if (counter % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0; // Indicates successful program execution.
}
