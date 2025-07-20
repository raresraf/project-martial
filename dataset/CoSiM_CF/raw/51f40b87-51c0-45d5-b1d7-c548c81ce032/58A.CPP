#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int a = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'h' && (a == 0))
      a++;
    else if ((s[i] == 'e') && (a == 1))
      a++;
    else if ((s[i] == 'l') && (a == 2))
      a++;
    else if ((s[i] == 'l') && (a == 3))
      a++;
    else if ((s[i] == 'o') && (a == 4))
      a++;
    else
      cout << "";
  }
  if (a == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}