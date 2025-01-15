#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  int cnt = 0;
  for (int i = 0; s[i] != '\0'; i++) {
    for (int j = i; s[j] != '\0'; j++) {
      if (s[i] == s[j + 1]) {
        s[j + 1] = '#';
        // cnt++;
      }
    }
  }
  cout << s << endl;
  int j = 0;
  char temp[100];
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] >= 'a' && s[i] <= 'z' || s[i] >= 'A' && s[i] <= 'Z') {
      temp[j] = s[i];
      j++;
    }
  }
  temp[j] = '\0';
  cout << temp << endl;
  int a;
  a = strlen(temp);
  if (a % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
