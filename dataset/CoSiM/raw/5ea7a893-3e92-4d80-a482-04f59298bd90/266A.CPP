#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  char d = s[0];
  int c = 0;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == d)
      c += 1;
    else
      d = s[i];
  }
  cout << c << endl;
}