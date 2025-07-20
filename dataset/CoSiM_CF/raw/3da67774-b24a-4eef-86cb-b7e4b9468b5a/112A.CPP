#include <bits/stdc++.h>

using namespace std;

int main() {
  string str1, str2;
  cin >> str1;
  cin >> str2;
  int i, size, x = 0;
  size = str1.length();
  for (i = 0; i < size; i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }
  if (str1 > str2) {
    cout << "1" << endl;
  } else if (str1 < str2) {
    cout << "-1" << endl;
  } else {
    cout << "0" << endl;
  }
  return 0;
}
