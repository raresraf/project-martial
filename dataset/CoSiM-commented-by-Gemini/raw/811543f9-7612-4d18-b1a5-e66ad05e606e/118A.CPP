/**
 * @file 118A.CPP
 * @brief Processes a string by deleting vowels and prepending dots to consonants.
 *
 * This program takes a string, converts it to lowercase, removes all vowels
 * ('a', 'o', 'y', 'e', 'u', 'i'), and inserts a '.' character before each
 * remaining character (consonants).
 *
 * @algorithm Multi-pass string transformation using vectors.
 * @complexity
 * Time: O(N^2), where N is the length of the string. The first pass to
 *       lowercase the string is O(N). The second pass to remove vowels is
 *       O(N^2) in the worst case because `vector::erase` is an O(N) operation.
 *       The final passes to build and print the result are linear with respect
 *       to the intermediate string length.
 * Space: O(N), for the two vectors (`myvector`, `myvector2`) used to store
 *        intermediate results.
 */
#include <algorithm>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  // C-style character array for input.
  char word[100];
  int i = 0;
  // A vector to hold the string after converting to lowercase.
  vector<char> myvector;
  // A vector to build the final output string.
  vector<char> myvector2;
  
  cin >> word;

  // --- Stage 1: Convert input to lowercase ---
  // Pre-condition: 'word' contains the null-terminated input.
  // Block Logic: Iterate through the input and store a lowercase version in myvector.
  while (word[i] != '\0') {
    myvector.push_back(tolower(word[i]));
    i++;
  }

  // --- Stage 2: Remove vowels from the vector ---
  // Pre-condition: myvector contains the full lowercase string.
  // Block Logic: Iterate through myvector and erase any character that is a vowel.
  for (i = 0; i < myvector.size(); i++) {
    if (myvector[i] == 'a' || myvector[i] == 'o' || myvector[i] == 'i' ||
        myvector[i] == 'y' || myvector[i] == 'e' || myvector[i] == 'u') {
      // vector::erase is an O(N) operation, making this loop O(N^2).
      myvector.erase(myvector.begin() + i);
      // Inline: Decrement 'i' to re-evaluate the character that shifted into
      // the current position after the erasure.
      i--;
    }
  }

  // --- Stage 3: Build the final output string ---
  // Pre-condition: myvector contains only consonants.
  // Block Logic: For each consonant, prepend a '.' and add both to myvector2.
  for (i = 0; i < myvector.size(); i++) {
    myvector2.push_back('.');
    myvector2.push_back(myvector[i]);
  }

  // --- Stage 4: Print the result ---
  for (i = 0; i < myvector2.size(); i++) {
    cout << myvector2[i];
  }
  
  return 0;
}
