#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  int cnt = 0;
  char a = s[0];
  for (int i = 1; i < n; i++) {
    if (a == s[i]) {
      cnt++;
    }
    a = s[i];
  }
  cout << cnt;
}