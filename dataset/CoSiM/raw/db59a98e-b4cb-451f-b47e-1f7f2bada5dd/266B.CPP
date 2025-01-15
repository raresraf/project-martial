#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  for (int a = 0; a < t; a++) {
    for (int i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }

  cout << s;
  return 0;
}