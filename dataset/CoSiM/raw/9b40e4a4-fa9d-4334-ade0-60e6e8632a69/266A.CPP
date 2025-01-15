#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, i, c = 0;
  cin >> n;
  string s;
  cin >> s;
  for (i = 1; i < n; i++) {
    if (s[i - 1] == s[i]) {
      c++;
    }
  }
  cout << c;
}
