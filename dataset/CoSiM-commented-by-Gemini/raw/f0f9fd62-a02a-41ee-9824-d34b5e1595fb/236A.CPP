/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct
 *        characters in their username.
 *
 * This program checks if the number of distinct characters in a given username
 * is even or odd. If it's even, the user is identified as female ("CHAT WITH HER!").
 * If it's odd, the user is male ("IGNORE HIM!").
 *
 * @b Algorithm: The program uses a frequency array `a` of size 26 to count the
 *    occurrences of each letter in the username. It then counts how many letters
 *    of the alphabet are *not* present in the username. The parity of this count
 *    of missing letters is used to determine the output. Since 26 is an even
 *    number, `26 - distinct_chars` has the same parity as `distinct_chars`, so
 *    this indirect logic works correctly.
 *
 * @b Time_Complexity: O(L), where L is the length of the username string, because
 *    it iterates through the string once. The second loop is constant time (26 iterations).
 * @b Space_Complexity: O(1), as the frequency array has a fixed size (26).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;
  int a[26] = {0}; // Frequency array for the 26 lowercase letters.
  int ctr = 0;     // Counter for missing characters.
  
  /**
   * @brief This loop populates the frequency array.
   * `str[i] - 'a'` calculates the 0-25 index for each character.
   */
  for (int i = 0; i < str.length(); i++)
    a[str[i] - 'a']++;
    
  /**
   * @brief This loop counts the number of characters that do not appear in the string.
   */
  for (int i = 0; i < 26; i++) {
    if (a[i] < 1)
      ++ctr;
  }
  
  /**
   * @brief Determine the output based on the parity of the missing characters count.
   * If `26 - distinct_chars` is even, `distinct_chars` must also be even.
   * If `26 - distinct_chars` is odd, `distinct_chars` must also be odd.
   */
  cout << (ctr % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!");
  return 0;
}