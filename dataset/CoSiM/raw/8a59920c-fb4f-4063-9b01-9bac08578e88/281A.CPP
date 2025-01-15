#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);
  string s;
  cin >> s;
  s[0] = toupper(s[0]);
  cout << s;
  return 0;
}
