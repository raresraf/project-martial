#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n, t;
  cin >> n >> t;

  string s;
  cin >> s;

  for (int j = 0; j < t; ++j) {
    bool is_boy = s[0] == 'B';
    for (int i = 1; i < s.size(); ++i) {
      if (s[i] == 'G') {
        if (is_boy)
          swap(s[i], s[i - 1]);
        is_boy = false;
      } else {
        is_boy = true;
      }
    }
  }
  cout << s << endl;
}
