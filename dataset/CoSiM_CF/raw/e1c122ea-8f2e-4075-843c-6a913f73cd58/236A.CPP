#include <bits/stdc++.h>
#include <string>
using namespace std;
int main() {
  string str;
  cin >> str;
  int l = str.length();
  for (int i = 0; i < l; i++) {
    for (int j = i + 1; j < l; j++) {
      if (str[i] == str[j]) {
        for (int k = j; k < l; k++) {
          str[k] = str[k + 1];
        }
        j--;
        l--;
      }
    }
  }
  if (l % 2 != 0) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }
}
