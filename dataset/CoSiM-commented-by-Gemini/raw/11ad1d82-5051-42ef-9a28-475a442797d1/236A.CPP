#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string s;
  int a[100], x = 0;
  cin >> s;
  for (int i = 0; i <= s.length(); i++) {
    if (s[i] != 0) {
      for (int j = i + 1; j <= s.length(); j++) {
        if (s[i] == s[j]) {
          s[j] = 0;
        }
      }
      x++;
    }
  }
  if (x % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
}
