/**
 * @file 96A.CPP
 * @brief Checks for a "dangerous" situation in a sequence of events.
 *
 * @details This program determines if there is a sequence of at least 7 consecutive
 * identical characters in a given input string. This is a common competitive
 * programming problem, often framed as players of a single team having possession
 * of a ball for 7 or more consecutive turns.
 *
 * Algorithm: Single-pass linear scan.
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(1) (as the string buffer is of a fixed size).
 */

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

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;

int main() {
  char str[100] = {0};
  int len = 0;
  int f = 0;       // Counter for consecutive identical characters.
  int flag = 0;  // Flag to indicate if a dangerous situation is found.

  // Functional Utility: Read a line from standard input.
  fgets(str, 100, stdin);
  len = strlen(str);
  
  // Block Logic: Sanitize input by removing the trailing newline character
  // that fgets might include.
  if (str[len - 1] == '
') { // Correctly checks for newline
    str[len - 1] = '\0';
    len--;
  }

  /**
   * Block Logic: Perform a linear scan to find consecutive characters.
   * Pre-condition: `str` contains the null-terminated input string.
   * Invariant: `f` holds the count of consecutive identical character pairs ending at `i`.
   */
  for (int i = 0; i < len; i++) {
    // Inline: If a character is the same as the next one, we extend the current run.
    if (str[i] == str[i + 1])
      f++;
    // Inline: If the sequence is broken, reset the counter.
    else
      f = 0;

    /**
     * Block Logic: Check for the dangerous condition.
     * A count of 6 consecutive pairs means a total of 7 identical characters in a row.
     */
    if (f == 6) {
      flag = 1;
      cout << "YES" << endl;
      break;
    }
  }

  // Block Logic: If the loop completes without finding a sequence of 7,
  // the situation is not dangerous.
  if (flag == 0) {
    cout << "NO" << endl; // Corrected to "NO" for consistency with "YES".
  }

  return 0;
}
