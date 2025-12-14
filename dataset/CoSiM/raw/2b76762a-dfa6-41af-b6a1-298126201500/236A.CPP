#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

char s[26];
int main() {
  char ch[105];
  cin >> ch;
  int cnt = 0;
  int len = strlen(ch);
  for (int i = 0; i < len; i++) {
    int t = (ch[i] - 96);
    if (!s[t]) {
      cnt++;
      s[t] = 1;
    }
  }
  if (cnt & 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
