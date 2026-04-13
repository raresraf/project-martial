/*#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
using namespace std;

int main(){
    char str[100]={0};
    int len=0,f=0;
        fgets(str,100,stdin);
        len=strlen(str);
        if(str[len-1]!='\0'){
            str[len-1]='\0';
            len--;
        }
        for(int i=0;i<len;i++){
            if(str[i]==str[i+1]){
                f++;
                if(f==6){
                    cout<<"YES"<<endl;
                    break;
                }
            }
            else f=1;
        }
        if(f<6) cout<<"NO"<<endl;
    return 0;
}

#include<cstdio>
#include<cstring>
#include<iostream>
#include<string>
using namespace std;

int main(){
    char str[100]={0};
    int len=0,f=0,c1=0,c2=0;
        fgets(str,100,stdin);
        len=strlen(str);
        if(str[len-1]!='\0'){
            str[len-1]='\0';
            len--;
        }
        for(int i=0;i<len;i++){
            if(str[i]=='0')
             c1++;


            else c2++;
        }
        if(c1==7 || c2==7)
         { f=1;
         }
if(f==1){
cout<<"yes"<<endl;}
else{
cout<<"no"<<endl;}
    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
    int flag, f;
    char s[105];
    scanf("%s", s);
    f=flag=0;
    for(int i=0; i<strlen(s); i++) {
        if(s[i]==s[i+1])
            f++;
        else f=0;
        if(f==6) {
            flag=1;
            printf("YES");
            break;
        }
    }
    if(flag==0) printf("NO");
    return 0;
}*/

/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem: "Football".
 *
 * This program determines if a situation in a football game is "dangerous".
 * A situation is considered dangerous if there are 7 or more players of the
 * same team (represented by '0' or '1') positioned consecutively in a row.
 *
 * Algorithm: Linear scan to count consecutive identical characters.
 * Time Complexity: O(L), where L is the length of the input string, as it
 *                  iterates through the string once.
 * Space Complexity: O(L) for storing the input string.
 */
#include <cstdio>   // Required for C-style input/output operations like fgets, strlen.
#include <cstring>  // Required for C-style string functions like strlen.
#include <iostream> // Required for C++ style output operations like cout, endl.
#include <string>   // Required for std::string (though char array is used for input).

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads a string representing player positions, counts consecutive identical
 * players, and determines if a "dangerous" situation (7 or more consecutive) exists.
 */
int main() {
  char str[100] = {0}; // Declares a C-style character array to store the input string, initialized to zeros.
  int len = 0;         // Stores the actual length of the string.
  int f = 0;           // Counter for consecutive identical characters. Initialized to 0.
  int flag = 0;        // Flag to indicate if a dangerous situation has been found (1 for YES, 0 for NO).

  fgets(str, 100, stdin); // Reads a line from standard input into 'str'. `fgets` includes the newline character.
  len = strlen(str);      // Calculates the length of the string, including the newline.

  /**
   * @brief Removes trailing newline character from `fgets` input.
   * Block Logic: If the last character read by `fgets` is not a null terminator,
   * it means a newline character was captured. This block replaces the newline
   * with a null terminator and decrements the length.
   */
  if (str[len - 1] != '\0') {
    str[len - 1] = '\0'; // Replace newline with null terminator.
    len--;               // Adjust the length.
  }

  /**
   * @brief Iterates through the string to count consecutive identical players.
   * Block Logic: Compares each character with its successor. If they are the same,
   * the consecutive counter `f` is incremented. If they differ, `f` is reset to 0.
   * If `f` reaches 6, it means 7 identical characters have been found consecutively.
   * Pre-condition: `str` contains the player sequence, `len` is its effective length.
   * Invariant: `f` tracks the current count of consecutive identical characters.
   */
  for (int i = 0; i < len; i++) {
    if (str[i] == str[i + 1]) // Checks if the current character is the same as the next character.
      f++;                     // If identical, increment the consecutive counter.
    else
      f = 0; // If different, reset the consecutive counter.

    if (f == 6) { // If 'f' reaches 6, it means 7 consecutive identical characters have been found.
                  // (e.g., if str[0] == str[1], f=1. If str[0] == str[1] == str[2], f=2. So f=6 means 7 characters.)
      flag = 1;         // Set the flag to indicate a dangerous situation.
      cout << "YES" << endl; // Print "YES".
      break;            // Exit the loop as the condition is met.
    }
  }

  /**
   * @brief Prints "No" if no dangerous situation was found.
   * Block Logic: After the loop completes, if `flag` is still 0, it means no
   * sequence of 7 or more consecutive identical players was found.
   */
  if (flag == 0) {
    cout << "No" << endl; // Print "No".
  }

  return 0; // Indicates successful program execution.
}
