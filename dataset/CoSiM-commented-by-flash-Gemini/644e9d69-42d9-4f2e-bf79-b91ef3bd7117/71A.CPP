#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s.size() > 10)
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << "\n";
    else
      cout << s << "\n";
  }
  return (0);
}