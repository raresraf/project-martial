#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  char ch;
  int n, i;
  cin >> str;
  n = str.size();
  ch = toupper(str[0]);
  cout << ch;
  for (i = 1; i < n; i++) {
    cout << str[i];
  }
  return 0;
}
