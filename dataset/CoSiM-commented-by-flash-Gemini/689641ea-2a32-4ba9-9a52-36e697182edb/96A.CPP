// Author : Mosiur Rahman Sweet

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int counter = 1;
  int flag = 0;
  for (int i = 1; s[i]; i++) {
    if (s[i] == s[i - 1])
      counter++;
    else
      counter = 1;

    if (counter == 7) {
      flag = 1;
      break;
    }
  }
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
