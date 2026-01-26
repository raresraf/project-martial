#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  int i, n, j, k, l;
  cin >> n;
  j = 0;
  for (i = 0; i < n; i++) {
    cin >> s;
    if (s[0] == 'X' && s[1] == '+') {
      j++;
    }
    if (s[0] == '+') {
      j++;
    }
    if (s[0] == 'X' && s[1] == '-') {
      j--;
    }
    if (s[0] == '-') {
      j--;
    }
  }
  cout << j << endl;
  return 0;
}
