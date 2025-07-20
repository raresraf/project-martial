#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, t, tmp = 1;
  string s;
  cin >> n >> t >> s;

  for (int k = 0; k < t; ++k) {
    for (int i = 1; i < n; ++i) {
      if (s[i - 1] == 'B' && s[i] == 'G') {
        swap(s[i - 1], s[i]);
        i++;
      }
      //   if( tmp == t) break;
    }
  }

  cout << s << "\n";

  return 0;
}
