/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @author chavodon
 *
 * @details This program processes a list of words. For each word, if its
 * length is greater than 10, it is abbreviated. The abbreviation consists
 * of the first letter, the number of letters between the first and last
 * letter, and the last letter. Otherwise, the word is printed as is.
 *
 * Link: https://codeforces.com/problemset/problem/71/A
 *
 * Algorithm:
 * 1. Read the number of words, 'n'.
 * 2. Read all 'n' words and concatenate them into a single string `longWord`,
 *    using '!' as a delimiter.
 * 3. Iterate through `longWord`, processing one word at a time between delimiters.
 * 4. For each word extracted:
 *    a. If its length is greater than 10, create an abbreviation:
 *       - Take the first character.
 *       - Convert the length minus 2 into a string.
 *       - Take the last character.
 *       - Concatenate these three parts.
 *    b. If the length is 10 or less, the word remains unchanged.
 * 5. Print the resulting (abbreviated or original) word.
 *
 * Time Complexity: O(L), where L is the total length of all words combined.
 * Space Complexity: O(L) to store the concatenated string of all words.
 */

#include <iostream>
#include <sstream>

using namespace std;

int main() {
  string word, longWord, newWord;
  int i, n, count = 0, letterCount;
  char f, l;
  cin >> n;
  // This loop reads all words and concatenates them into a single string,
  // delimited by '!'. This is an unconventional approach.
  for (i = 0; i < n; i++) {
    cin >> word;
    longWord += word + '!';
  }
  
  // This loop processes the concatenated string.
  while (longWord[count] != '\0') {
    letterCount = 0;
    newWord = "";
    f = longWord[count];
    // This inner loop extracts one word at a time from the concatenated string.
    while (longWord[count + letterCount] != '!') {
      newWord += longWord[count + letterCount];
      letterCount++;
    }
    // Move the main counter to the beginning of the next word.
    count += letterCount + 1;
    // Get the last character of the extracted word.
    l = longWord[count - 2];
    
    // Check if the word's length requires abbreviation.
    if (letterCount > 10) {
      // Use stringstream to convert the number of middle characters to a string.
      stringstream ss;
      ss << letterCount - 2;
      string s;
      ss >> s;
      // Construct the abbreviated word.
      newWord = f + s + l;
    }
    cout << newWord << endl;
  }
  return 0;
}
