/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 236A - "Boy or Girl").
 *
 * This program determines a gender-like classification ("Boy" or "Girl") for a username
 * based on the parity of the number of unique characters it contains.
 * If the number of unique characters is even, the output is "CHAT WITH HER!";
 * if odd, the output is "IGNORE HIM!".
 *
 * Algorithm: Character frequency counting and parity check.
 * It utilizes a fixed-size array (`check[26]`) to keep track of encountered lowercase
 * English alphabet characters. It iterates through the input string, marks the presence
 * of each character, and increments a counter for unique characters. Finally, it checks
 * the parity of this unique character count.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * The string is traversed once.
 * Space Complexity: O(1), because the `check` array size is constant (26 for the English alphabet),
 * regardless of the input string length.
 */
#include <cstdio> // Required for C-style input/output operations (scanf, printf).

using namespace std;

int check[26]; // Global array to mark the presence of unique lowercase characters 'a' through 'z'.
               // Index 0 for 'a', 1 for 'b', ..., 25 for 'z'.

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a username, counts its unique characters using a boolean array, and prints a message
 * based on the parity of the unique character count.
 */
int main() {
  int n = 0;       // Counter for the number of unique characters found.
  int i, temp;     // Loop counter 'i', 'temp' to store character's alphabet index.
  char x[101];     // Character array to store the input username (max 100 chars + null terminator).
  // Pre-condition: Read the input username string. Assumes lowercase input or handles conversion implicitly.
  scanf("%s", &x); // Reads a string until whitespace.
  // Block Logic: Iterate through each character of the input string.
  // Invariant: Loop continues until the null terminator (0) is encountered.
  for (i = 0; x[i] != 0; i++) {
    temp = x[i] - 'a'; // Functional Utility: Convert character to 0-indexed alphabet position.
    // Block Logic: Check if this character has been encountered before.
    if (check[temp] == 1) {
      continue; // If seen, skip to the next character.
    } else {
      check[temp] = 1; // Mark the character as seen.
      n++;             // Increment the count of unique characters.
    }
  }
  // Block Logic: Check the parity of the unique character count.
  if (n % 2 == 0)      // If the number of unique characters is even.
    printf("CHAT WITH HER!"); // Print message for "Girl".
  else                 // If the number of unique characters is odd.
    printf("IGNORE HIM!"); // Print message for "Boy".
  return 0; // Indicate successful execution.
}