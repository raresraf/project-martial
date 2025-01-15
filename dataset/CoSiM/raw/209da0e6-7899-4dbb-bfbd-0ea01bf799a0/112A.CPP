#include <bits/stdc++.h>
using namespace std;

int main() {
  string str1, str2;
  cin >> str1 >> str2;
  int lowUp = 0;
  // int upLow=1;
  int same = 0;
  for (int i = 0; i < str1.length(); i++) {
    if (str1[i] >= 65 && str1[i] <= 91) {
      str1[i] = str1[i] + 32;
    }
  }
  for (int i = 0; i < str2.length(); i++) {
    if (str2[i] >= 65 && str2[i] <= 91) {
      str2[i] = str2[i] + 32;
    }
  }
  if (str1.compare(str2) == 0) {
    cout << same << endl;
  } else {
    for (int i = 0; i < str1.length(); i++) {
      if (str1[i] < str2[i]) {
        lowUp = 0;
        break;
      } else if (str1[i] == str2[i]) {
        if (str1[i + 1] < str2[i + 1]) {
          lowUp = 0;
          break;
        }
      } else {
        lowUp++;
        break;
      }
    }
    if (lowUp != 0) {
      cout << "1" << endl;
    } else {
      cout << "-1" << endl;
    }
  }
  return 0;
}
