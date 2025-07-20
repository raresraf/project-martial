#define _CRT_SECURE_NO_WARNINGS
#include <bits/stdc++.h>

using namespace std;

int main() {
  int len;
  cin >> len;
  char *p = (char *)malloc(sizeof(char) * len);
  cin >> p;
  char tmp;
  tmp = *p;
  int count = 1;
  while (*p++ && *p != '\0') {
    if (*p != tmp) {
      count++;
      tmp = *p;
    }
  }
  int ret = len - count;
  cout << ret;
  return 0;
}