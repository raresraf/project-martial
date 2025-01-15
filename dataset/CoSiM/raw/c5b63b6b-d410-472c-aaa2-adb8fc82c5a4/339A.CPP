#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int a[51] = {0}, i, j = 0;
  string s;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (i % 2 == 0) {
      a[j] = s[i] - 48;
      j++;
    }
  }
  sort(a, a + j);
  j = 0;
  for (i = 0; i < s.length(); i++) {
    if (i % 2 == 0) {
      printf("%d", a[j]);
      j++;
    } else
      printf("+");
  }
}
