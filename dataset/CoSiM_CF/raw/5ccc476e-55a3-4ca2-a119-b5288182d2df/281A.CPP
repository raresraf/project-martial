#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int upper = toupper(s[0]);
  if (s[0] == upper) {
    cout << s << '\n';
  } else {
    s[0] = upper;
    cout << s << '\n';
  }
  return 0;
}