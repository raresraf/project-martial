#include <bits/stdc++.h>
using namespace std;
int main() {

  string str;
  cin >> str;
  char ch[] = {'h', 'e', 'l', 'l', 'o'};
  for (int i = str.length() - 1, j = 4; i >= 0; i--) {
    if (str[i] == ch[j])
      j--;
    if (j < 0) {
      cout << "YES\n";
      return 0;
    }
  }

  cout << "NO\n";
  return 0;
}