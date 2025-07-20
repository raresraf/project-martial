// g++  7.4.0

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  char *s = new char[n + 1];
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }
  int cnt = 0;
  for (int i = 0; i < n - 1; i++) {

    if (s[i] == s[i + 1]) {
      cnt++;
    }
  }
  cout << cnt << endl;
  return 0;
}
