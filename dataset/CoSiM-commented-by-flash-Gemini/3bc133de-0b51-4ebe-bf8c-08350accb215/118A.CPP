/**
 * @file 118A.CPP
 * @brief This program processes an input string by transforming it according to specific rules:
 * 1. All vowels (a, o, y, e, u, i, case-insensitive) are removed.
 * 2. All consonants are converted to lowercase.
 * 3. A '.' character is inserted before each consonant.
 * This is a classic string manipulation problem, often found in competitive programming.
 * Algorithm: Linear scan with character classification and string concatenation.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass and string building.
 * Space Complexity: O(N) for the resulting string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

using namespace std;
#define EPS (1e-7) // A small epsilon value, often used for floating-point comparisons.
#define INF (1e9) // A large integer value, often used to represent infinity.
#define PI (acos(-1)) // Defines PI using the arccosine of -1.

typedef long long ll; // Defines 'll' as a shorthand for 'long long'.
#define endl '\n' // Redefines 'endl' to a newline character for faster output.

int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1}; // Arrays for directional movements (e.g., in grid problems).

const int MOD = 1000000007; // A modulo constant, often used in problems involving large numbers and modular arithmetic.

//////////////////////////////////////////

void Input() {} // Placeholder function for input reading, not used in this specific problem.

void Solve() {} // Placeholder function for solving logic, not used in this specific problem.

/**
 * @brief Checks if a given character is a vowel (case-insensitive, including 'y').
 *
 * @param c The character to check.
 * @return True if the character is a vowel, False otherwise.
 */
bool isvowel(char c) {

  // Conditional Logic: Checks if the character 'c' matches any of the uppercase or lowercase vowels (including 'y').
  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'A' ||
      c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'y' || c == 'Y') {

    return true; // Returns true if it's a vowel.
  }

  return false; // Returns false if it's not a vowel.
}

int main() {

  // Functional Utility: Unties cin from cout and disables synchronization with C standard I/O.
  // This speeds up input/output operations, crucial for competitive programming.
  cin.tie(0);
  ios::sync_with_stdio(false);
  Input(); // Calls the placeholder Input function.
  Solve(); // Calls the placeholder Solve function.

  string s; // Declares a string variable 's' to store the input.
  cin >> s; // Reads the input string from standard input.

  string ans = ""; // Initializes an empty string to build the processed result.

  int len = s.size(); // Stores the length of the input string for loop control.

  /**
   * Block Logic: Iterates through each character of the input string 's'.
   * It processes each character: removing vowels, converting consonants to lowercase,
   * and prepending a '.' to each consonant in the resulting string.
   */
  for (int i = 0; i < len; i++) {

    // Conditional Logic: Checks if the current character is NOT a vowel.
    if (!isvowel(s[i])) {

      // Conditional Logic: Checks if the consonant is already lowercase.
      // ASCII values: 'a' (97) to 'z' (122).
      if (s[i] >= 97 && s[i] <= 122) {

        ans += '.'; // Prepends a '.' to the result string.
        ans += s[i]; // Appends the lowercase consonant.
      }
      // Conditional Logic: Checks if the consonant is uppercase and converts it to lowercase.
      // ASCII values: 'A' (65) to 'Z' (90). Adding 32 converts uppercase to lowercase.
      if (s[i] >= 65 && s[i] <= 90) {

        ans += '.'; // Prepends a '.' to the result string.
        ans += s[i] + 32; // Appends the lowercase version of the consonant.
      }
    }
  }

  cout << ans << endl; // Prints the final processed string.

  return 0; // Indicates successful program execution.
}
