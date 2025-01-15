
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;

  cin >> s;

  for (int i = 0; i < s.size(); i = i + 2) {
    for (int j = 0; j < s.size() - i - 1; j = j + 2) {
      if (s[j] > s[j + 2])
        swap(s[j + 2], s[j]);
    }
  }
  for (int j = 0; j < s.size(); j++)
    cout << s[j];

  return 0;
}
