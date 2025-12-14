// 266A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  string s;
  cin >> n;
  cin >> s;
  char a;
  int count = 0;
  a = s[0];
  for (int i = 1; i < s.length(); i++) {
    if (a == s[i]) {
      count++;
    } else
      a = s[i];
  }
  cout << count;
  return 0;
}
