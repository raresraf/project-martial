#include <bits/stdc++.h>
using namespace std;
int main() {
  int cou = 0;
  char s[100];
  cin >> s;
  for (int i = 0; i < strlen(s); i++) {
    if (s[i] == '4' || s[i] == '7')
      cou++;
  }
  if (cou == 4 || cou == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
