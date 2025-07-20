#include <bits/stdc++.h>

using namespace std;

char *captword(char *str) {
  if (*str > 90) {
    *str -= 32;
  }
  return str;
}

int main() {
  char *p = (char *)malloc(sizeof(char) * 1000);
  cin >> p;
  captword(p);
  cout << p;
  return 0;
}