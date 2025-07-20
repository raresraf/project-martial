#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  s[0] = s[0] >= 97 ? s[0] - 32 : s[0];
  cout << s;
  return 0;
}