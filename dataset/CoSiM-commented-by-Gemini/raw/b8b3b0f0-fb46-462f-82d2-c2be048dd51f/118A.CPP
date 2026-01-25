#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[101];
  char temp[101];
  int j = 0;
  cin >> s;
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] != 'A' && s[i] != 'a' && s[i] != 'E' && s[i] != 'e' &&
        s[i] != 'I' && s[i] != 'i' && s[i] != 'O' && s[i] != 'o' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'Y') {
      temp[j] = s[i];
      j++;
    }
  }
  temp[j] = '\0';
  // cout<<temp;
  for (int i = 0; temp[i] != '\0'; i++) {
    if (temp[i] >= 'A' && temp[i] <= 'Z') {
      temp[i] = temp[i] + 32;
    }
  }
  for (int i = 0; temp[i] != '\0'; i++) {
    cout << "." << temp[i];
  }
}
