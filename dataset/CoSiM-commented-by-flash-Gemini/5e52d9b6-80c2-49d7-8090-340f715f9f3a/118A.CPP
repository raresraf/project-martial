/**
 * @5e52d9b6-80c2-49d7-8090-340f715f9f3a/118A.CPP
 * @brief This program attempts to solve problem 118A ("String Task").
 * The problem requires transforming a word by converting uppercase letters to lowercase,
 * removing all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and then printing each remaining
 * consonant prefixed with a dot ('.').
 *
 * NOTE: The provided code contains unused function definitions and commented-out
 * sections in `main`, suggesting an incomplete or evolving solution. The active
 * part of `main` calls helper functions that perform the transformation and printing.
 * The `delvovel` function itself prints partial output.
 *
 * Algorithm (Active part): Two-step processing: case conversion, then vowel removal and formatted printing.
 * Time Complexity: O(L) where L is the length of the input string, due to linear passes.
 * Space Complexity: O(L) for storing strings.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
#include <cctype>   // For tolower, but custom logic is used in ctos.
using namespace std; // Uses the standard namespace.

/**
 * @brief Converts all uppercase letters in a string to lowercase.
 * This function iterates through the input string and modifies characters in-place
 * if they are uppercase letters.
 * @param a The input string to convert.
 * @return The modified string with all letters in lowercase.
 * Time Complexity: O(L) where L is the length of the string.
 * Space Complexity: O(1) if modification is considered in-place.
 */
string ctos(string a) {
  int len, i;
  i = a.length(); // Gets the length of the string.
  /**
   * Block Logic: Iterates through the string to convert uppercase characters to lowercase.
   * Invariant: After this loop, all characters that were uppercase are now lowercase.
   */
  for (len = 0; len < i; len++) { // 'len' is used as loop counter here.
    // Inline: Checks if the character is an uppercase letter (ASCII 'A' through 'Z').
    if (a[len] <= 'Z' && a[len] >= 'A')
      a[len] = a[len] + 32; // Inline: Converts uppercase to lowercase by adding 32 to ASCII value.
  }
  return a; // Returns the modified string.
}

/**
 * @brief Deletes specified vowels from a string and prints the remaining consonants formatted.
 * This function iterates through the input string (assumed to be lowercase),
 * filters out vowels ('a', 'e', 'i', 'o', 'u', 'y'), constructs a new string with
 * only consonants, and then prints these consonants prefixed with a dot.
 * It also prints the raw consonant string first, which might be unintended intermediate output.
 * @param a The input string (expected to be lowercase).
 * @return The string containing only consonants.
 * Time Complexity: O(L) where L is the length of the string.
 * Space Complexity: O(L) for the new consonant string.
 */
string delvovel(string a) {
  int len, i, j, k, count; // Declares variables: len, i, j, k (unused), count (consonant count).
  string ans; // Declares a string to build the consonant-only result.
  len = a.length(); // Gets the length of the input string.
  count = 0; // Initializes consonant counter.
  /**
   * Block Logic: Iterates through the input string 'a', filtering out vowels and
   * appending consonants to 'ans'.
   * Invariant: 'ans' contains only consonants from 'a', and 'count' is their total number.
   */
  for (i = 0; i < len; i++) { // Loops through each character of the string.
    // Inline: Checks if the character is NOT one of the specified lowercase vowels.
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      ans[count] = a[i]; // Appends the consonant to 'ans'. (Note: direct assignment to `ans[count]` on empty `ans` is unsafe, typically `ans += a[i]` or `ans.push_back(a[i])` is used).
      count++;             // Increments consonant count.
    }
  }
  // This line is problematic: `ans` is a `std::string`, not a `char*`. Assigning '\0' this way is incorrect.
  // The intended use of ans[count] implies ans is treated like a char array, which `std::string` is not.
  ans[count] = '\0'; 
  cout << ans << endl; // Prints the raw consonant string, potentially with garbage characters after null terminator depending on previous operation.
  /**
   * Block Logic: Prints each consonant from the 'ans' string, prefixed with a dot.
   * Invariant: The formatted consonant string is output.
   */
  for (i = 0; i < count; i++) { // Loops through the collected consonants.
    cout << '.'; // Prints a dot.
    cout << ans[i]; // Prints the consonant.
  }
  //	ans=ans-count; // Commented out: This line appears to be an incomplete or incorrect operation.
  cout << endl; // Prints a newline character.
  return ans; // Returns the consonant string.
}
/*char * insert(string a)
{
        int i,j,len;
        char *ans;
        len=a.length();
        for(i=0;i<2*len;i++)
        {
                if(i%2==0)
                        ans[i]='.';
                else
                        ans[i]=a[i/2];
        }
        ans[2*len]='\0';
        return ans;
}*/
int main() {
  string test;      // Declares a string variable for input.
  int i, len, j;    // Declares variables: i, len, j (unused).
  char ans[200];    // Declares a C-style char array (unused in active code).
  cin >> test;      // Reads the input string.
  test = ctos(test); // Converts the input string to lowercase.
  test = delvovel(test); // Removes vowels and prints formatted output.
  // cout << test << endl; // Commented out: Would print the (potentially problematic) consonant string.
  /*	int new_var=0;
          len=test.length();
          for(i=0;i<len;i++)
          {
                  if(test[i]!='\0')
                          new_var++;
          }

          for(i=0;i<2*new_var;i++)
          {
                  j=i/2;
                  if(i%2==0)
                          ans[i]='.';
                  else
                          ans[i]=test[j];
          }
  //	cout << len << endl;
          ans[2*new_var]='\0';
          if(ans[2*new_var-1]=='.')
                  ans[2*new_var-1]='\0';
          cout << ans << endl;
          return 0;*/
  // The program implicitly returns 0 on successful completion of main.
}
