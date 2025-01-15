#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  cin >> str;

  int n = str.length();

  for (int i = 0; i < n; i++) {
    str[i] = tolower(str[i]);
  }

  for (int i = 0; i < n; i++) {
    if (str[i] != 'a' && str[i] != 'e' && str[i] != 'i' && str[i] != 'o' &&
        str[i] != 'u' && str[i] != 'y')
      cout << "." << str[i];
  }
}
