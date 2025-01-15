#include <bits/stdc++.h>
using namespace std;
int main() {
  int count = 1;
  string str;
  cin >> str;
  transform(str.begin(), str.end(), str.begin(), ::tolower);
  for (int i = 0; i < str.length(); i++) {
    if (!(str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
          str[i] == 'u' || str[i] == 'y')) {
      cout << "." << str[i];
    }
  }
}
