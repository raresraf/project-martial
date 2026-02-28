/**
 * @file 118A.CPP
 * @brief A flawed and buggy attempt to solve the "String Task" problem from Codeforces.
 * @details The goal is to delete vowels, insert a '.' before each consonant, and
 * convert all consonants to lowercase. This solution tries to use functions to
 * separate the logic, but contains significant implementation errors, including
 * undefined behavior.
 *
 * @algorithm Flawed Multi-Function Approach
 * 1. `ctos`: A function to convert the input string to lowercase. This part works correctly.
 * 2. `delvowel`: A function intended to remove vowels and print the result. This
 *    function is critically flawed. It attempts to build a new string by assigning
 *    characters to indices of an empty `std::string`, which is undefined behavior.
 *    The actual printing logic is a side effect within this function and operates on
 *    this improperly constructed string.
 * 3. `main`: The main function orchestrates the calls to `ctos` and `delvowel`.
 *
 * @complexity
 * - Intended Time: O(L), where L is the length of the string.
 * - Actual Behavior: Unpredictable due to undefined behavior, but likely still O(L).
 * - Space: O(L) for storing strings.
 */
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Converts a string to lowercase.
 * @param a The input string.
 * @return The lowercase version of the string.
 */
string ctos(string a) {
  int len, i;
  i = a.length();
  for (len = 0; len < i; len++) {
    // Manually convert uppercase characters to lowercase using ASCII offset.
    if (a[len] <= 'Z' && a[len] >= 'A')
      a[len] = a[len] + 32;
  }
  return a;
}

/**
 * @brief Attempts to delete vowels and print the formatted consonant string.
 * @warning This function has critical bugs and relies on undefined behavior.
 * @param a The input string (expected to be lowercase).
 * @return A potentially corrupted or empty string. The function's useful output
 *         is printed directly to `cout` as a side effect.
 */
string delvovel(string a) {
  int len, i, j, k, count;
  string ans; // `ans` is an empty string.
  len = a.length();
  count = 0;
  for (i = 0; i < len; i++) {
    // Check if the character is NOT a vowel.
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      
      // *** UNDEFINED BEHAVIOR ***
      // Writing to an out-of-bounds index of an empty std::string.
      // The correct way to append a character is `ans.push_back(a[i])` or `ans += a[i]`.
      ans[count] = a[i];
      count++;
    }
  }
  
  // Manually adding a null terminator is unnecessary for std::string and suggests
  // confusion with C-style strings.
  ans[count] = '\0';
  
  // This output is unreliable due to the undefined behavior above.
  cout << ans << endl;
  
  /**
   * @brief This loop produces the intended output format.
   * It iterates through the (likely corrupted) `ans` string and prints each
   * character prefixed with a dot. This is the code's "real" output.
   */
  for (i = 0; i < count; i++) {
    cout << '.';
    cout << ans[i];
  }
  
  cout << endl;
  return ans; // The returned string is not used in main and is not correctly built.
}

/* This is an abandoned attempt at a function to insert dots.
char * insert(string a)
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
  string test;
  int i, len, j;
  char ans[200]; // Unused C-style string.
  
  cin >> test;
  
  // First, convert the entire string to lowercase.
  test = ctos(test);
  
  // Second, call the flawed function to process and print the result.
  // The return value is assigned but immediately discarded. The program's
  // output is generated entirely as a side effect inside `delvovel`.
  test = delvovel(test);
  
  /* This block is another abandoned attempt at the logic, also commented out.
  // cout << test << endl;
  	int new_var=0;
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
}
