/**
 * @8bc2536d-308c-4c6b-b0bb-c0c5feb441d5/118A.CPP
 * @brief This program processes an input string by attempting to remove vowels
 * (both uppercase and lowercase) and then printing the remaining characters
 * (consonants), each prefixed with a dot. All consonants are converted to lowercase.
 * Note: The vowel removal logic contains a known flaw where modifying the vector
 * while iterating can lead to incorrect results or crashes (indicated by "// not working for char 'u'").
 * Algorithm: String manipulation, character-by-character processing with flawed in-place removal.
 * Time Complexity: Potentially worse than O(N) due to repeated vector erasures in a loop,
 * where N is the length of the input string. Each `erase` operation can be O(N) itself.
 * Space Complexity: O(N) for storing character vectors.
 */

#include <bits/stdc++.h> // Includes various standard libraries, including iostream, string, vector, algorithm.
using namespace std;

#define vi vector<int> // Macro for vector of integers (not used for its intended type in this specific code).
#define pb push_back   // Macro for push_back operation.

/**
 * @brief Processes the input string to remove vowels and format consonants.
 * This function iterates through a string, attempts to remove vowels, converts remaining
 * characters to lowercase, and then prefixes them with a dot.
 * @param s The input string to be processed.
 * @return A new string with vowels removed and consonants formatted.
 */
// not working for char 'u' - Known issue: Modifying the vector while iterating
// using `erase` invalidates iterators and affects loop control.
string resultString(string s) {
  // Inline: Convert input string to a vector of characters for easier manipulation.
  vector<char> arr(s.begin(), s.end());

  // Block Logic: Iterate through the character vector to identify and remove vowels.
  // Invariant: The vector 'arr' is modified in-place by removing vowel characters.
  // Pre-condition: 'arr' contains characters from the input string.
  for (size_t i = 0; i < arr.size(); i++) { // Flaw: arr.size() changes during iteration due to erase().
    // Block Logic: Check if the current character is a vowel (case-insensitive) using a switch statement.
    switch (arr[i]) { // Flaw: arr[i] may be invalid after erase operations.
    case 'a':
      arr.erase(find(arr.begin(), arr.end(), arr[i])); // Flaw: find returns iterator, erase shifts elements.
      break;
    case 'e':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'i':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'o':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'u':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'y':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'A':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'E':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'I':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'O':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'U':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'Y':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    default:
      // Inline: If not a vowel, continue to the next character without modification.
      continue;
      break;
    }
  }
  
  // Block Logic: Build a new vector of characters with '.' prefixes and lowercase consonants.
  vector<char> arr2;
  // Invariant: Each consonant from 'arr' is converted to lowercase and prefixed with a dot in 'arr2'.
  // Pre-condition: 'arr' contains the (possibly incorrectly) filtered consonants.
  for (size_t i = 0; i < arr.size(); i++) {
    arr2.pb('.'); // Append a dot.
    arr2.pb(tolower(arr[i])); // Append the lowercase version of the character.
  }
  
  // Inline: Convert the processed character vector back to a string.
  string p(arr2.begin(), arr2.end());
  return p; // Return the final processed string.
}

/**
 * @brief Main function for program execution.
 * Reads an input string, processes it using resultString, and prints the result.
 * @return 0 upon successful execution.
 */
int main() {
  // @var s: Stores the input string.
  string s;
  cin >> s; // Read the input string.
  // Block Logic: Call resultString to process the input and print the returned string.
  cout << resultString(s) << "
"; // Print the processed string followed by a newline.
  return 0; // Explicitly return 0 for successful execution.
}
