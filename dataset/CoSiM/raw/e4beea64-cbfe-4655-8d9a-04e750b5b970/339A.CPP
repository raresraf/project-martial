#include <bits/stdc++.h>

using namespace std;

int main() {
  string str1;
  int i;
  cin >> str1;
  sort(str1.begin(), str1.end());
  str1.erase(remove(str1.begin(), str1.end(), '+'), str1.end());
  auto size = str1.length();
  for (i = 0; i < size; i++) {
    cout << str1[i];
    if (i != size - 1) {
      cout << '+';
    }
  }
  return 0;
}
