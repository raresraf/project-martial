#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  char ch = str[0];
  if (ch >= 97)
    str[0] = ch - 32;
  cout << str;
}
