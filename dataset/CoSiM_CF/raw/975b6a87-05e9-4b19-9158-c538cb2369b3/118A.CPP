
/// Bismillah.

#include <bits/stdc++.h>
using namespace std;
int main() {
  char str[101], con[101];

  int len, i, j;
  // cout<<"Enter a string : ";

  // gets(str);
  cin >> str;

  len = strlen(str);

  int v = 0, c = 0;

  for (i = 0; i < len; i++) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'I' || str[i] == 'O' || str[i] == 'U' || str[i] == 'Y') {
      // vowel[v] = str[i];
      // v++;
    } else {
      con[c] = str[i];
      c++;
    }
  }
  // vowel[v]='\0';
  con[c] = '\0';

  for (int i = 0; i < strlen(con); i++) {
    if (con[i] > 'A' && con[i] < 'Z') {
      con[i] += 32;
    }

    cout << "." << con[i];
  }
  return 0;
}
