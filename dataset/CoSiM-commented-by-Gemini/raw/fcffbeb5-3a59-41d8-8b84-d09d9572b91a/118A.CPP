/**
 * @file 118A.cpp
 * @brief Solution to a competitive programming problem: String Task.
 * This program processes a string according to a specific set of rules:
 * 1. Convert all uppercase letters to lowercase.
 * 2. Delete all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 3. Insert a "." character before each consonant.
 *
 * Algorithm:
 * 1. Read a line of text into a character array.
 * 2. Iterate through the array:
 *    a. Convert any uppercase letters to lowercase.
 *    b. Check if the character is a vowel.
 *    c. If it's not a vowel (i.e., it's a consonant), print a "." followed by the character.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  char arr[101]; // Buffer to hold the input string.
  int i, j;
  
  // Read a line of input, up to 100 characters.
  cin.getline(arr, 101);
  int len = strlen(arr);
  
  // Iterate through the string to process each character.
  for (i = 0; i < len; i++) {
    // Convert uppercase letters to lowercase using ASCII values.
    if (arr[i] >= 'A' && arr[i] <= 'Z') {
      arr[i] = arr[i] + 32; // The difference between 'a' and 'A' is 32.
    }
    
    // Check if the character is a consonant (not a vowel).
    if (arr[i] != 'a' && arr[i] != 'e' && arr[i] != 'i' && arr[i] != 'o' &&
        arr[i] != 'u' && arr[i] != 'y') {
      // If it's a consonant, print a "." followed by the character.
      cout << "." << arr[i];
    }
  }
  cout << endl;
}
