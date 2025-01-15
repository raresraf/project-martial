#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  string s;
  cin >> s;
  int prev = s.at(0);
  int timeout = 6;
  for (int i = 1; i < s.length(); i++) {
    int cur = s.at(i);
    if (cur == prev) {
      timeout--;
      if (timeout == 0) {
        printf("YES\n");
        return 0;
      }
    } else {
      timeout = 6;
    }
    prev = cur;
  }
  printf("NO\n");
  return 0;
}
