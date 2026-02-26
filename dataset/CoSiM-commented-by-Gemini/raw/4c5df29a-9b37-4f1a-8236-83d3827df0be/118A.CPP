/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The problem requires processing a string according to three rules:
 * 1. Convert all uppercase letters to their lowercase equivalents.
 * 2. Delete all vowels. In this problem, the vowels are 'a', 'e', 'i', 'o', 'u', and 'y'.
 * 3. Before each consonant, insert a '.' character.
 *
 * @algorithm
 * The program iterates through the input string character by character in a single pass.
 * For each character, it first converts it to lowercase if it's uppercase.
 * Then, it checks if the resulting character is a consonant (i.e., not a vowel).
 * If it is a consonant, it prints a '.' followed by the character itself.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string.
 * (Note: The use of `strlen(s)` inside the loop condition makes the theoretical
 * complexity O(N^2), but for small N (<=100) it's not an issue. A more
 * optimal approach would be to calculate the length once before the loop.)
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
#include <cstring>
#include <cstdio>

using namespace std;

int main() {
  int i;
  char s[101]; 
  cin >> s;

  /**
   * @block
   * @brief Iterates through the string to process each character.
   */
  for (i = 0; i < strlen(s); i++) {
    /**
     * @block
     * @brief Convert uppercase letters to lowercase.
     * @logic It checks if the character's ASCII value is in the uppercase range.
     * If so, it converts it to lowercase using ASCII arithmetic.
     */
    if (s[i] >= 'A' && s[i] <= 'Z') {
      s[i] = s[i] + ('a' - 'A'); 
    }

    /**
     * @block
     * @brief Process consonants.
     * @logic It checks if the (now lowercase) character is NOT a vowel.
     * If it's a consonant, it prints a '.' followed by the character.
     */
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      printf(".%c", s[i]);
    }
  }
  printf("\n"); 
  return 0;
}
