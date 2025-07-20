#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  char s[n];
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    cout << s[i];
  }
}