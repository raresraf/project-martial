/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details The problem requires a case-insensitive lexicographical comparison
 * of two strings. The program should output -1, 0, or 1 based on whether the
 * first string is lexicographically smaller than, equal to, or greater than
 * the second.
 *
 * @algorithm
 * The code performs a character-by-character comparison of the two strings.
 * 1. Read two C-style strings from input.
 * 2. Iterate through both strings simultaneously.
 * 3. In each iteration, convert the current characters from both strings to
 *    lowercase to ensure case-insensitivity.
 * 4. If the lowercase characters are different, the lexicographical order is
 *    determined. Set a result variable to 1 or -1 and break the loop.
 * 5. If the characters are the same, continue to the next pair.
 * 6. If the loop completes without finding any differences, the strings are
 *    considered equal. Set the result variable to 0.
 * 7. Print the final result.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings, as it involves
 * a single pass.
 * Space Complexity: O(L_max) for the two character array buffers.
 */
#include <bits/stdc++.h>
#include <cstdio>
using namespace std;

int main() {
  // C-style strings to hold the input.
  char str[101], str2[101];
  int i;
  // 'b' will store the comparison result (-1, 0, or 1).
  // Note: 'b' is uninitialized, which could be an issue if the loop never runs.
  int b;
  
  scanf("%s", str);
  scanf("%s", str2);
  
  /**
   * @block
   * @brief Compare strings character by character, ignoring case.
   */
  for (i = 0; str[i]; i++) {
    // Compare the lowercase versions of the current characters.
    if (tolower(str[i]) == tolower(str2[i])) {
      // If they are equal, the strings are equal so far.
      b = 0;
      continue;
    } 
    // This 'else if' is redundant, as this block is only reached if the
    // characters are already known to be different.
    else if (tolower(str[i]) != tolower(str2[i])) {
      if (tolower(str[i]) > tolower(str2[i])) {
        b = 1;
        break; // First difference found, result is 1.
      } else if (tolower(str[i]) < tolower(str2[i])) {
        b = -1;
        break; // First difference found, result is -1.
      }
    }
  }

  // Print the final result based on the value of 'b'.
  if (b == 0) {
    printf("0\n");
  } else if (b == 1) {
    printf("1\n");
  } else if (b == -1) {
    printf("-1\n");
  }
}

/*
 * @note The file contains a second, commented-out implementation of the same problem below.
 *
#include <cstdio>
#include <iostream>
#include <string>

#include <cstring>
using namespace std;
char a1[110];
char a2[110];
char b1[110];
char b2[110];


int Cmp(char *pstr1,char *pstr2)
  {int i,t;
  for(i=0;pstr1[i]!='\0';i++)
  if(pstr1[i]>='A'&&pstr1[i]<='Z')
  *(pstr1+i)=*(pstr1+i)+32;
  for(i=0;pstr2[i]!='\0';i++)
  if(pstr2[i]>='A'&&pstr2[i]<='Z')
  *(pstr2+i)=*(pstr2+i)+32;
  t=0;
  for(i=0;i<strlen(pstr1);i++)
      {if(*(pstr1+i)>*(pstr2+i))
              {t=1;break;}
        if(*(pstr1+i)<*(pstr2+i))
              {t=-1;break;}
      if(*(pstr1+i)>*(pstr2+i))
      continue;

      }
        return t;
  }



int main()
  {int t;
  gets(a1);
  gets(a2);
  t=Cmp(a1,a2);

  if(t==1) cout<<1;
  if(t==-1)cout<<-1;
  if(t==0)cout<<0;
      cin.get();
          cout<<endl;

return 0;
}
*/
