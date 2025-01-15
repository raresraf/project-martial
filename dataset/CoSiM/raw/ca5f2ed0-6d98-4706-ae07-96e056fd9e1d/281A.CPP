/*
   Link to problem here
   https://codeforces.com/problemset/problem/281/A
*/
#include <iostream>
using namespace std;
int main() {
  string s;
  getline(cin, s);
  int i;
  if ((int)s[0] >= 97 && (int)s[0] <= 122) {
    s[0] -= 32;
  }
  cout << s;
  return 0;
}