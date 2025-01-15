#include <iostream>
using namespace std;
int main() {
  string s;
  int n, t;
  cin >> n >> t;
  cin >> s;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j++;
      }
    }
  }
  cout << s;
  return 0;
}
