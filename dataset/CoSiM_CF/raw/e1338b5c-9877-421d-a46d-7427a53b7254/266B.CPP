#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s.at(i) == 'B' && s.at(i + 1) == 'G') {
        s.at(i) = 'G';
        s.at(i + 1) = 'B';
        i += 1;
      } else
        continue;
    }
  }

  cout << s;
  return 0;
}
