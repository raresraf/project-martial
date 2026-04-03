/**
 * @8e1e63ad-04ab-4503-a64c-62ee63a15c3a/71A.CPP
 * @brief This program implements word abbreviation for "Way Too Long Words" (Codeforces 71A).
 * It reads a list of words and, for each word, if its length exceeds a predefined
 * threshold (10 characters), it abbreviates the word to its first character,
 * the count of characters between the first and last, and its last character.
 * Otherwise, the word is printed without modification.
 * Algorithm: String length check and conditional formatting.
 * Time Complexity: O(N * L), where N is the number of words and L is the average length of words.
 * Space Complexity: O(L) for storing the current word.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.
using namespace std;

int main() {
  // @var threshold: Constant defining the maximum length for a word to be printed in full.
  const int threshold = 10;
  // @var numWords: The total number of words to be processed.
  int numWords;
  cin >> numWords; // Read the total number of words.
  
  // Block Logic: Process each word until all 'numWords' have been handled.
  // Invariant: 'numWords' decreases with each iteration until all words are processed.
  // Pre-condition: 'numWords' indicates how many words to read and process.
  do {
    // @var input: Stores the current word read from standard input.
    string input;
    cin >> input; // Read the current word.
    
    // @var inputSize: Stores the length of the current input word.
    int inputSize = input.size();
    
    // Block Logic: Check if the word's length is within the threshold for full printing.
    if (inputSize <= threshold) {
      cout << input << endl; // Print the word as is.
    } 
    // Block Logic: If the word's length exceeds the threshold, abbreviate it.
    else {
      // Inline: Print the first character, then the count of intermediate characters, then the last character.
      cout << input[0] << inputSize - 2 << input[inputSize - 1] << endl;
    }
  } while (--numWords); // Decrement 'numWords' and continue loop until it becomes zero.

  return 0; // Explicitly return 0 for successful execution.
}
