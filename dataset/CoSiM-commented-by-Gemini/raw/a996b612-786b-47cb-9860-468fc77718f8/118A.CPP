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
 * @algorithm
 * 1. A helper function `isVowel` is defined to check if a character is one of the vowels.
 * 2. The main function reads the input string.
 * 3. It iterates through each character of the string.
 * 4. For each character, it's first converted to lowercase.
 * 5. `isVowel` is called to check the character.
 * 6. If the character is not a vowel (i.e., it's a consonant), a '.' followed by the
 *    lowercase character is printed directly to standard output.
 *
 * This approach is memory-efficient as it avoids creating a new result string.
 *
 * @time_complexity O(L), where L is the length of the input string. The `isVowel`
 *                  function takes constant time as it checks a fixed-size array.
 * @space_complexity O(L) to store the input string.
 */
#include <iostream>
#include <string>
#include <cctype> // For tolower()

using namespace std;

// Global array of vowel characters for lookup.
char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};

/**
 * @brief Checks if a character is a vowel.
 * @param c The character to check.
 * @return True if the character is in the global `vowels` array, false otherwise.
 */
bool isVowel(char c) {
  for (int i = 0; i < 6; i++) {
    if (c == vowels[i]) {
      return true;
    }
  }
  return false;
}

int main() {
  string s;
  cin >> s;
  
  /**
   * @brief Main loop to process the input string.
   */
  for (int i = 0; i < s.length(); i++) {
    // Convert character to lowercase for case-insensitive comparison.
    s[i] = tolower(s[i]);
    
    // If the character is not a vowel, it's a consonant.
    if (!isVowel(s[i])) {
      // Print the required format: "." followed by the consonant.
      cout << "." << s[i];
    }
  }
  cout << "
";
  
  return 0;
}
