/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem (Codeforces 118A).
 * @author competitive-programmer
 *
 * @details This program transforms a string based on a set of rules:
 * 1. Convert all uppercase letters to lowercase.
 * 2. Remove all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 3. Insert a '.' character before each remaining character (consonant).
 *
 * Algorithm:
 * 1. Read the input string into a character array `A`.
 * 2. Create a new character array `B` to store the result.
 * 3. Iterate through each character of `A`.
 * 4. For each character, convert it to lowercase.
 * 5. Check if the character is a vowel.
 * 6. If it's a consonant, append a '.' to `B`, followed by the consonant itself.
 * 7. After the loop, null-terminate the result string `B` and print it.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief A custom implementation to find the length of a C-style string.
 * @param A The input character array.
 * @return The length of the string, stopping at a newline, null terminator, or space.
 */
int strlen(char A[]) {
  int i = 0;
  while (A[i] != '\n' && A[i] != '\0' && A[i] != ' ')
    i++;
  return i;
}

/**
 * @brief Performs the main string transformation logic.
 * @param A The input character array.
 */
void solve(char A[]) {
  int n = strlen(A);
  // `B` is a buffer to build the output string.
  char B[1000];
  // `c` is the index for the output buffer `B`.
  int c = 0;
  
  /**
   * Block Logic: Iterate through each character of the input string `A`.
   */
  for (int i = 0; i < n; i++) {
    // Convert the character to lowercase if it's uppercase.
    if (A[i] - 'A' >= 0 && A[i] - 'Z' <= 0) {
      A[i] = A[i] - 'A' + 'a';
    }
    
    // Pre-condition: Check if the character is NOT a vowel.
    if (A[i] != 'a' && A[i] != 'e' && A[i] != 'i' && A[i] != 'o' &&
        A[i] != 'u' && A[i] != 'y') {
      // If it's a consonant, add a '.' and the consonant to the result buffer.
      B[c] = '.';
      c++;
      B[c] = A[i];
      c++;
    }
  }
  // Null-terminate the resulting C-style string.
  B[c] = '\0';
  printf("%s\n", B);
}

int main() {
  char A[101];

  // Read the input string.
  scanf("%s", A);

  // Process the string.
  solve(A);

  return 0;
}
