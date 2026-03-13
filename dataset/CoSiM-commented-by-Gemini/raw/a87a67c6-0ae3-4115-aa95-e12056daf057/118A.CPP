/**
 * @file 118A.CPP
 * @brief Solution for CodeForces Problem 118A - "String Task".
 *
 * @details
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * @algorithm Direct String Traversal and Printing
 * 1. Read the input string.
 * 2. Iterate through each character of the string using a range-based for loop.
 * 3. For each character, convert it to its lowercase equivalent.
 * 4. Check if the lowercase character is a consonant (i.e., not one of 'a', 'e', 'i', 'o', 'u', 'y').
 * 5. If it is a consonant, print a '.' followed by the character directly to standard output.
 *
 * This approach is memory-efficient as it avoids creating a new result string in memory.
 *
 * @time_complexity O(L), where L is the length of the input string.
 * @space_complexity O(L) to store the input string. The output is streamed, using O(1) additional space.
 */
#include <iostream>
#include <string>
#include <cctype> // For tolower()

using namespace std;

int main() {
  string word;
  cin >> word;

  /**
   * @brief Main loop to iterate through the input string and print the transformed output.
   * This uses a C++11 range-based for loop for concise iteration.
   */
  for (char c : word) {
    // Convert character to lowercase for case-insensitive vowel checking.
    c = tolower(c);

    // Check if the character is NOT a vowel.
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y') {
      // If it's a consonant, print a '.' followed by the lowercase character.
      cout << "." << c;
    }
  }
  
  return 0;
}
