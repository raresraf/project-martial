#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  for (int i = 0; i < a.length(); i++) {
    char ac = a.at(i);
    char bc = b.at(i);
    if (ac != bc) {
      if (ac > bc) {
        printf("1\n");
      } else {
        printf("-1\n");
      }
      return 0;
    }
  }
  printf("0\n");
  return 0;
}
