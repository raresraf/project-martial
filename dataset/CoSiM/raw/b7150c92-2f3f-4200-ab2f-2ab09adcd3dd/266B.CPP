#include <iostream>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  for (int j = 0; j < t; ++j)
    for (int i = 0; i < s.size() - 1;)
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i += 2;
      } else
        ++i;
  cout << s;
}