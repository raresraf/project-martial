#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[105];
  int ln, i;
  cin >> s;
  ln = strlen(s);
  for (i = 0; i < ln; i++) {
    /*while(1)
    {
    if(s[i]=='A' || s[i]=='a' || s[i]=='O' || s[i]=='o' || s[i]=='Y' ||
    s[i]=='y' || s[i]=='E ' || s[i]=='e' || s[i]=='U' || s[i]=='u' || s[i]=='I '
    || s[i]=='i')
    {
        i++;
    }
    else
        break;
    }*/
    if (s[i] >= 'A' && s[i] <= 'Z' && s[i] != 'A' && s[i] != 'O' &&
        s[i] != 'Y' && s[i] != 'E' && s[i] != 'U' && s[i] != 'I') {
      char c = s[i] + 32;
      cout << "." << c;
    } else if (s[i] >= 'a' && s[i] <= 'z' && s[i] != 'a' && s[i] != 'o' &&
               s[i] != 'y' && s[i] != 'e' && s[i] != 'u' && s[i] != 'i') {
      cout << "." << s[i];
    }
  }
  cout << endl;
}
