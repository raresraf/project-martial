//
// Created by Taro Young on 2020/9/7.
//
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  char s1[110], s2[110];
  int ans = 0;
  cin >> s1;
  cin >> s2;
  for (int i = 0; i < strlen(s1); i++) {
    int ch1 = tolower(s1[i]);
    int ch2 = tolower(s2[i]);
    if (ch1 < ch2) {
      ans = -1;
      break;
    }
    if (ch1 > ch2) {
      ans = 1;
      break;
    }
  }
  cout << ans;
  return 0;
}
