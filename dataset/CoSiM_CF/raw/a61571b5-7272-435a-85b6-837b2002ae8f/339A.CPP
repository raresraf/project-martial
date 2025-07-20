#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n[3] = {0}, i = 0;
  cin >> s;
  while (i < s.length()) {
    n[s[i] - '1']++;
    i += 2;
  }
  for (i = 0; i < 3; i++) {
    if (n[i]--) {
      cout << i + 1;
      break;
    }
  }
  for (; i < 3; i++) {
    while (n[i]--)
      cout << "+" << i + 1;
  }
  return 0;
}