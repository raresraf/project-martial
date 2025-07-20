#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, t, i, j;
  cin >> n >> t;
  string s;
  cin >> s;
  for (i = 0; i < t; i++) {
    j = 1;
    while (j < n) {
      if (s[j - 1] == 'B' && s[j] == 'G') {
        swap(s[j - 1], s[j]);
        j += 2;
      } else {
        j++;
      }
    }
  }
  cout << s;
}
