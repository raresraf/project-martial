#include <iostream>
#include <string>
using namespace std;
int main() {
  int x = 0, i, n;
  string s[150];
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> s[i];
    if (s[i][1] == '+')
      x++;
    else if (s[i][1] == '-')
      x--;
  }
  cout << x;
  return 0;
}
