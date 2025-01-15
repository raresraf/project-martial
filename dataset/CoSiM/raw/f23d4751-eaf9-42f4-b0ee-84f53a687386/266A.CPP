#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int i = 0, n = 0, p;
  cin >> p;
  cin >> s;
  p = s[i++];
  while (i < s.length()) {
    if (p == s[i])
      n++;
    else
      p = s[i];
    i++;
  }
  cout << n;
  return 0;
}